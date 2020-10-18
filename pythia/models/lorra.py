# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import sys
from models.Attention2 import *
from models.module_net import *

from utils.utils import unique_columns

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from pythia.modules.layers import ClassifierLayer


@registry.register_model("lorra")
class LoRRA(Pythia):
    def __init__(self, config, num_vocab_txt, num_vocab_nmn, out_num_choices,
                embed_dim_nmn, embed_dim_txt, image_height, image_width, in_image_dim,
                hidden_size, assembler, layout_criterion, answer_criterion,max_layout_len, num_layers=1, decoder_dropout=0,**kwarg):
        super().__init__(config)

        self.assembler = assembler
        self.layout_criterion = layout_criterion
        self.answer_criterion = answer_criterion

        ##initatiate attentionSeq2seq
        mySeq2seq = attention_seq2seq(myEncoder, myDecoder)
        self.mySeq2seq = mySeq2seq.cuda() if use_cuda else mySeq2seq

        ##initiate moduleNet
        myModuleNet = module_net(image_height=image_height, image_width=image_width, in_image_dim=in_image_dim,
                                 in_text_dim=embed_dim_txt, out_num_choices=out_num_choices, map_dim=hidden_size)

        self.myModuleNet = myModuleNet.cuda() if use_cuda else myModuleNet

    def build(self):
        self._init_text_embeddings("text")
        # For LoRRA context feature and text embeddings would be identity
        # but to keep a unified API, we will init them also
        # and we need to build them first before building pythia's other
        # modules as some of the modules require context attributes to be set
        self._init_text_embeddings("context")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")
        super().build()

    def get_optimizer_parameters(self, config):
        params = super().get_optimizer_parameters(config)
        params += [
            {"params": self.context_feature_embeddings_list.parameters()},
            {"params": self.context_embeddings.parameters()},
            {"params": self.context_feature_encoders.parameters()},
        ]

        return params

    def _get_classifier_input_dim(self):
        # Now, the classifier's input will be cat of image and context based
        # features
        return 2 * super()._get_classifier_input_dim()

    def forward(self,  input_txt_variable, input_text_seq_lens,
                input_images, input_answers,
                input_layout_variable, sample_list, policy_gradient_baseline=None,
                baseline_decay=None):
        sample_list.text = self.word_embedding(sample_list.text)
        #text_embedding_total = self.process_text_embedding(sample_list)

        ##run attentionSeq2Seq
        myLayouts, myAttentions, neg_entropy, log_seq_prob = \
            self.mySeq2seq(input_txt_variable, input_text_seq_lens, input_layout_variable,sample_list)

        layout_loss = None
        if input_layout_variable is not None:
            layout_loss = torch.mean(-log_seq_prob)

        predicted_layouts = np.asarray(myLayouts.cpu().data.numpy())
        expr_list, expr_validity_array = self.assembler.assemble(predicted_layouts)

        ## group samples based on layout
        sample_groups_by_layout = unique_columns(predicted_layouts)

        ##run moduleNet
        answer_losses = None
        policy_gradient_losses = None
        avg_answer_loss =None
        total_loss = None
        updated_baseline = policy_gradient_baseline
        current_answer = np.zeros(batch_size)

        for sample_group in sample_groups_by_layout:
            if sample_group.shape == 0:
                continue

            first_in_group = sample_group[0]
            if expr_validity_array[first_in_group]:
                layout_exp = expr_list[first_in_group]

                if input_answers is None:
                    ith_answer_variable = None
                else:
                    ith_answer = input_answers[sample_group]
                    ith_answer_variable = Variable(torch.LongTensor(ith_answer))
                    ith_answer_variable = ith_answer_variable.cuda() if use_cuda else ith_answer_variable

                textAttention = myAttentions[sample_group, :]

                ith_image = input_images[sample_group, :, :, :]
                ith_images_variable = Variable(torch.FloatTensor(ith_image))
                ith_images_variable = ith_images_variable.cuda() if use_cuda else ith_images_variable

                ##image[batch_size, H_feat, W_feat, D_feat] ==> [batch_size, D_feat, W_feat, H_feat] for conv2d
                #ith_images_variable = ith_images_variable.permute(0, 3, 1, 2)

                ith_images_variable = ith_images_variable.contiguous()

                myAnswers = self.myModuleNet(input_image_variable=ith_images_variable,
                                        input_text_attention_variable=textAttention,
                                        target_answer_variable=ith_answer_variable,
                                        expr_list=layout_exp)
                current_answer[sample_group] = torch.topk(myAnswers, 1)[1].cpu().data.numpy()[:, 0]

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        context_embedding_total, _ = self.process_feature_embedding(
            "context", sample_list, text_embedding_total, ["order_vectors"]
        )


        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            ["text", "text"],
            [myAnswers, context_embedding_total],
        )

        scores = self.calculate_logits(joint_embedding)

        return {"scores": scores}

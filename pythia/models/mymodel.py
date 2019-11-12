import torch

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from .utils import grad_mul_const # mask_softmax, grad_reverse, grad_reverse_mask, 
from pythia.modules.layers import ClassifierLayer
from block.models.networks.mlp import MLP


@registry.register_model("MyModel")
class MyMODEL(Pythia):
    def __init__(self, config, classif):
        super().__init__(config)
        self.c_1 = MLP(**classif)


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


    def forward(self, sample_list):

        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        context_embedding_total, _ = self.process_feature_embedding(
            "context", sample_list, text_embedding_total, ["order_vectors"]
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            ["image", "text"],
            [image_embedding_total, text_embedding_total, context_embedding_total],
        )
		"""
        - logits: the original predictions of the model
        - logits_q: the predictions from the question-only branch
        - logits_rubi: the updated predictions from the model by the mask.
    	=> Use `logits_rubi` and `logits_q` for the loss
    	"""

		logits = self.calculate_logits(joint_embedding)
		q_embedding = grad_mul_const(text_embedding_total, 0.0) # don't backpropagate through question encoder
		q_pred = self.c_1(q_embedding)
		scores = logits * torch.sigmoid(q_pred)

        return {"scores": scores}



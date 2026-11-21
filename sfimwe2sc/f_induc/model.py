import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from sfimwe2sc.f_induc.dataset import MODEL_INPUT_KEYS


def select_target(hidden_states, target_tidx):
    """各系列の `target_tidx` 番目のベクトルを取り出す。"""
    return hidden_states[torch.arange(len(target_tidx)), target_tidx]


class BaseNet(nn.Module):
    def __init__(self, pretrained_model_name, normalization, device, layer=-1):
        super().__init__()
        config = AutoConfig.from_pretrained(
            pretrained_model_name, output_hidden_states=True
        )
        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_model_name, config=config
        ).to(device)

        self.normalization = normalization
        self.layer = layer
        self.device = device

    def forward(self, inputs):
        model_inputs = {
            k: inputs[k].to(self.device) for k in MODEL_INPUT_KEYS if k in inputs
        }
        hidden_states = self.pretrained_model(**model_inputs)["hidden_states"]
        embeddings = select_target(hidden_states[self.layer], inputs["target_tidx"])
        if self.normalization == "true":
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

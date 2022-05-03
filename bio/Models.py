import torch.nn as nn
from transformers import RobertaModel, DebertaModel


class roBERTa_classifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, output_attentions=False,
                                                    output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)

    def forward(self, x, att_mask, return_features=False):
        features = self.roberta(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        x = self.classifier(x)
        if return_features:
            return x, features[:, 0, :]
        else:
            return x

class deBERTa_classifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.roberta = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False, output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)

    def forward(self, x, att_mask, return_features=False):
        features = self.roberta(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        x = self.classifier(x)
        if return_features:
            return x, features[:, 0, :]
        else:
            return x

import torch.nn as nn
import torch

class GTT_FFN(nn.Module):
    def __init__(self, window_size, embedding_size, rep_dim=None, pre_trained_weights=None):
        super().__init__()
        self.rep_dim = window_size // (2 * 2) if rep_dim is None else rep_dim
        self.fc = nn.Linear(embedding_size, self.rep_dim, bias=False)

        # If pre-trained model are provided, initialize the layer with it
        if pre_trained_weights is not None:
            with torch.no_grad():
                if rep_dim is None:
                    raise ValueError("rep_dim should be specified if pretrained weights are loaded")
                self.fc.weight = nn.Parameter(pre_trained_weights)                

    def forward(self, x):
        x = self.fc(x)
        return x

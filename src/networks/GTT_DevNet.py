import torch
import torch.nn as nn
import torch.nn.functional as F

class GTT_DevNet(nn.Module):
    def __init__(self, window_size, embedding_size, topk_ratio = 0.1, rep_dim = None, pre_trained_weights = None):
        super().__init__()
        self.embedding_size = embedding_size
        self.topk = int(embedding_size * topk_ratio)
        self.rep_dim = window_size // (2 * 2) if rep_dim is None else rep_dim
        self.fc = nn.Linear(embedding_size, self.rep_dim, bias=False)

        # If pre-trained model are provided, initialize the layer with it
        if pre_trained_weights is not None:
            with torch.no_grad():
                if rep_dim is None:
                    raise ValueError("rep_dim should be specified if pretrained weights are loaded")
                self.fc.weight = nn.Parameter(pre_trained_weights)     

    def forward(self, embedding):
        scores = self.fc(embedding)
        if self.topk != self.embedding_size:
            scores = scores.view(int(scores.size(0)), -1)
            scores = torch.topk(torch.abs(scores), self.topk, dim=1)[0]
            scores = torch.mean(scores, dim=1).view(-1, 1)
        else:
            scores = scores.view(int(scores.size(0)), -1)
            scores = torch.mean(scores, dim=1).view(-1, 1)

        return scores

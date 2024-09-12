import torch.nn as nn

class GTT_FLOS(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()        
        self.rep_dim = 1  # Output embedding size
        self.fc = nn.Linear(embedding_size, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x
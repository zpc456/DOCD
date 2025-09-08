import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=2, size_average=True, device='cpu'):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.size_average = size_average

        if alpha is None:
            self.alpha = torch.ones(num_classes).to(device)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha).float().to(device)
        else:
            self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)

    def forward(self, logits, labels):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1))

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.view(-1, 1))

        alpha = self.alpha.gather(0, labels)

        loss = -alpha * (1 - pt).pow(self.gamma) * pt.log()

        return loss.mean() if self.size_average else loss.sum()

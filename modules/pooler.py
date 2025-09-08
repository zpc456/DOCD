import torch
from torch import nn


class Pooler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class AttentionPooling(nn.Module):
    def __init__(self,
                 config):
        super(AttentionPooling, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self,
                input_tensor,
                pooling_mask):
        pooling_score = self.linear1(input_tensor)
        pooling_score = self.activation(pooling_score)
        pooling_score = self.linear2(pooling_score)

        pooling_score += pooling_mask
        attention_probs = nn.Softmax(dim=1)(pooling_score)

        attention_output = (attention_probs * input_tensor).sum(dim=1)
        return attention_output

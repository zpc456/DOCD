import torch
from torch import nn
from modules.guide_attention import Attention


class IntermediateLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)
        self.activation = nn.ReLU()
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class OutputLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class GCTLayer(nn.Module):
    def __init__(self, args, stack_idx):
        super().__init__()
        self.attention = Attention(args, stack_idx)

    def forward(self, hidden_states, attention_mask=None, guide_mask=None, prior=None, output_attentions=True):
        self_attention_outputs = self.attention(hidden_states, attention_mask, guide_mask, prior, output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        outputs = (attention_output,) + outputs
        return outputs

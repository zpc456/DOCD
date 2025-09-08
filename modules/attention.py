import torch
from torch import nn
import math


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_attention_heads = args.num_heads
        self.attention_head_size = int(args.hidden_size / args.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class SelfOutput(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, context_layer, original_input_tensor):
        context_layer = self.dense(context_layer)
        context_layer = self.activation(context_layer)
        context_layer = self.dropout(context_layer)
        hidden_states = self.layer_norm(context_layer + original_input_tensor)

        return hidden_states


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attention = SelfAttention(args)
        self.self_output = SelfOutput(args)

    def forward(self, hidden_states, attention_mask, output_attentions=True):
        self_attention_outputs = self.self_attention(hidden_states, attention_mask, output_attentions)
        attention_output = self.self_output(self_attention_outputs[0], hidden_states)
        outputs = (attention_output,) + self_attention_outputs[1:]
        return outputs


class DAGAttention(nn.Module):
    def __init__(self, args):
        super(DAGAttention, self).__init__()
        self.self_attention = SelfAttention(args)
        self.self_output = SelfOutput(args)

        self.self_attention_dag = SelfAttention(args)
        self.self_output_dag = SelfOutput(args)

    def forward(self, input_tensor, input_tensor_dag, attention_mask, output_attentions=False):
        self_output = self.self_attention(input_tensor, attention_mask, output_attentions)
        attention_output = self.self_output(self_output[0], input_tensor)

        self_output_dag = self.self_attention_dag(input_tensor_dag, attention_mask, output_attentions)
        attention_output_dag = self.self_output_dag(self_output_dag[0], input_tensor_dag)
        outputs = (attention_output, attention_output_dag,) + self_output[1:]
        return outputs

import torch
from torch import nn
from modules.attention import DAGAttention


class DAGIntermediate(nn.Module):
    def __init__(self, args):
        super(DAGIntermediate, self).__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)
        self.dense_dag = nn.Linear(args.hidden_size, args.intermediate_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, hidden_states_dag):
        hidden_states_ = self.dense(hidden_states)
        hidden_states_dag_ = self.dense_dag(hidden_states_dag.float())
        hidden_states = self.activation(hidden_states_ + hidden_states_dag_)

        return hidden_states


class DAGOutput(nn.Module):
    def __init__(self, args):
        super(DAGOutput, self).__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_size)
        self.dense_dag = nn.Linear(args.intermediate_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size)
        self.LayerNorm_dag = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, hidden_states_, input_tensor, input_tensor_dag):
        hidden_states = self.dense(hidden_states_)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states_dag = self.dense_dag(hidden_states_)
        hidden_states_dag = self.dropout(hidden_states_dag)
        hidden_states_dag = self.LayerNorm_dag(hidden_states_dag + input_tensor_dag)

        return hidden_states, hidden_states_dag


class DAGEncoderLayer(nn.Module):
    def __init__(self, args):
        super(DAGEncoderLayer, self).__init__()
        self.attention = DAGAttention(args)
        self.intermediate = DAGIntermediate(args)
        self.output = DAGOutput(args)

    def forward(self, hidden_states, hidden_states_dag, attention_mask, output_attentions=False):
        attention_output, attention_output_dag, outputs = self.attention(hidden_states, hidden_states_dag, attention_mask, output_attentions)
        intermediate_output = self.intermediate(attention_output, attention_output_dag)
        layer_output, layer_output_dag = self.output(intermediate_output, attention_output, attention_output_dag)
        return layer_output, layer_output_dag, outputs


class DagEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = nn.ModuleList([DAGEncoderLayer(args) for _ in range(6)])

    def forward(self, hidden_states, dag_inputs, attention_mask, output_hidden_states=True, output_attentions=False):
        all_hidden_states = ()
        all_hidden_states_dag = ()
        all_attentions = () if output_attentions else None
        for layer_module in self.layer:
            hidden_states, dag_inputs, attentions = layer_module(hidden_states, dag_inputs, attention_mask, output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attentions,)
            if output_hidden_states:
                all_hidden_states = all_hidden_states+(hidden_states,)
                all_hidden_states_dag = all_hidden_states_dag + (dag_inputs,)
        if not output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_hidden_states_dag = all_hidden_states_dag + (dag_inputs,)

        return all_hidden_states, all_hidden_states_dag, all_attentions

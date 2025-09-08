import torch
from torch import nn
from modules.pooler import AttentionPooling
from modules.DAG_encoder import DagEncoder

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


class DAGAttention2D(nn.Module):
    def __init__(self, in_features, attention_dim_size):
        super(DAGAttention2D, self).__init__()
        self.attention_dim_size = attention_dim_size
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, attention_dim_size)
        self.linear2 = nn.Linear(attention_dim_size, 1)

    def forward(self, leaves, ancestors, mask=None):
        mask = mask.unsqueeze(2)
        x = torch.cat((leaves * mask, ancestors * mask), dim=-1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        mask_attn = (1.0 - mask) * VERY_NEGATIVE_NUMBER
        x = x + mask_attn
        x = torch.softmax(x, dim=1)
        x = (x * ancestors * mask).sum(dim=1)
        return x


class DxKnowledgeEncoder(nn.Module):
    def __init__(self, args):
        super(DxKnowledgeEncoder, self).__init__()
        self.args = args
        self.embed_dag = None

        self.dag_attention = DAGAttention2D(2 * args.hidden_size, args.hidden_size)

        self.encoder = DagEncoder(args)

        self.pooling_visit = AttentionPooling(args)
        self.pooling_dag = AttentionPooling(args)

        self.embed_init = nn.Embedding(args.dx_num_tree_nodes + 1, args.hidden_size,padding_idx=args.dx_num_tree_nodes)
        self.embed_inputs = nn.Embedding(args.dx_code_size + 1, args.hidden_size,padding_idx=args.dx_code_size)

    def forward(self, input_ids, code_mask=None, output_attentions=False):
        leaves_emb = self.embed_init(self.args.dx_leaves_list)
        ancestors_emb = self.embed_init(self.args.dx_ancestors_list)
        dag_emb = self.dag_attention(leaves_emb, ancestors_emb, self.args.dx_masks_list)
        padding = torch.zeros([1, self.args.hidden_size], dtype=torch.float32).to(self.args.device)
        dict_matrix = torch.cat([padding, dag_emb], dim=0)
        self.embed_dag = nn.Embedding.from_pretrained(dict_matrix, freeze=True)

        input_ids = input_ids.long()
        inputs = self.embed_inputs(input_ids)

        inputs_dag = self.embed_dag(input_ids)
        inputs_mask = code_mask
        extended_attention_mask = inputs_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER
        visit_outputs, dag_outputs, all_attentions = self.encoder(inputs, inputs_dag, extended_attention_mask, output_hidden_states=True, output_attentions=output_attentions)

        attention_mask = inputs_mask.unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * VERY_NEGATIVE_NUMBER

        visit_outs = visit_outputs[-1]
        dag_outs = self.pooling_dag(dag_outputs[-1], attention_mask)

        return visit_outs, dag_outs, all_attentions


class ProcKnowledgeEncoder(nn.Module):
    def __init__(self, args):
        super(ProcKnowledgeEncoder, self).__init__()
        self.args = args
        self.embed_dag = None

        self.dag_attention = DAGAttention2D(2 * args.hidden_size, args.hidden_size)

        self.encoder = DagEncoder(args)

        self.pooling_visit = AttentionPooling(args)
        self.pooling_dag = AttentionPooling(args)

        self.embed_init = nn.Embedding(args.proc_num_tree_nodes + 1, args.hidden_size,padding_idx=args.proc_num_tree_nodes)
        self.embed_inputs = nn.Embedding(args.proc_code_size + 1, args.hidden_size,padding_idx=args.proc_code_size)

    def forward(self, input_ids, code_mask=None, output_attentions=False):
        leaves_emb = self.embed_init(self.args.proc_leaves_list)
        ancestors_emb = self.embed_init(self.args.proc_ancestors_list)
        dag_emb = self.dag_attention(leaves_emb, ancestors_emb, self.args.proc_masks_list)
        padding = torch.zeros([1, self.args.hidden_size], dtype=torch.float32).to(self.args.device)
        dict_matrix = torch.cat([padding, dag_emb], dim=0)
        self.embed_dag = nn.Embedding.from_pretrained(dict_matrix, freeze=False)

        input_ids = input_ids.long()
        inputs = self.embed_inputs(input_ids)

        inputs_dag = self.embed_dag(input_ids)
        inputs_mask = code_mask
        extended_attention_mask = inputs_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER
        visit_outputs, dag_outputs, all_attentions = self.encoder(inputs, inputs_dag, extended_attention_mask, output_hidden_states=True, output_attentions=output_attentions)

        attention_mask = inputs_mask.unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * VERY_NEGATIVE_NUMBER

        visit_outs = visit_outputs[-1]
        dag_outs = self.pooling_dag(dag_outputs[-1], attention_mask)

        return visit_outs, dag_outs, all_attentions

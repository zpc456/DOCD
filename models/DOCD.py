import numpy as np
import os
import sys
import math
import torch
from torch import nn
from utils import get_extended_attention_mask
from modules.embedder import FeatureEmbedder
from modules.pooler import Pooler
from modules.DOCD_encoder_layer import DOCDLayer
from modules.KnowledgeEncoder import DxKnowledgeEncoder, ProcKnowledgeEncoder
from modules.focal_loss import FocalLoss


class DOCD(nn.Module):
    def __init__(self, args):
        super(DOCD, self).__init__()
        self.num_labels = args.num_labels
        self.dx_num_ccs_cat1 = args.dx_num_ccs_cat1
        self.proc_num_ccs_cat1 = args.proc_num_ccs_cat1
        self.reg_coef = args.reg_coef
        self.use_guide = args.use_guide
        self.use_prior = args.use_prior
        self.prior_scalar = args.prior_scalar
        self.batch_size = args.batch_size
        self.num_stacks = args.num_stacks
        self.max_num_codes = args.max_num_codes
        self.output_attentions = args.output_attentions
        self.output_hidden_states = args.output_hidden_states
        self.feature_keys = args.feature_keys
        self.layers = nn.ModuleList([DOCDLayer(args, i) for i in range(args.num_stacks)])
        self.embeddings = FeatureEmbedder(args)
        self.pooler = Pooler(args)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(args.hidden_size, args.num_labels)
        self.classifier_dx_ccs_cat1 = nn.Linear(args.hidden_size, args.dx_num_ccs_cat1)
        self.classifier_proc_ccs_cat1 = nn.Linear(args.hidden_size, args.proc_num_ccs_cat1)
        self.dx_knowledge_encoder = DxKnowledgeEncoder(args)
        self.proc_knowledge_encoder = ProcKnowledgeEncoder(args)
        self.loss_coef = args.loss_coef

    def create_matrix_vdp(self, features, masks, priors):
        batch_size = features['dx_ints'].shape[0]
        device = features['dx_ints'].device
        num_visit = 1
        num_dx_ids = self.max_num_codes if self.use_prior else features['dx_ints'].shape[-1]
        num_proc_ids = self.max_num_codes if self.use_prior else features['proc_ints'].shape[-1]
        num_demographics = features['demographics_ints'].shape[-1]
        num_vital_signs = features['vital_signs_ints'].shape[-1]
        num_static = num_demographics + num_vital_signs
        num_codes = num_visit + num_dx_ids + num_proc_ids + num_demographics + num_vital_signs

        guide = None
        if self.use_guide:
            row0 = torch.cat([torch.zeros([num_visit, num_visit]),
                              torch.ones([num_visit, num_demographics + num_vital_signs + num_dx_ids]),
                              torch.zeros([num_visit, num_proc_ids]), ], axis=1)
            row1 = torch.zeros([num_demographics, num_codes])
            row2 = torch.zeros([num_vital_signs, num_codes])
            row3 = torch.cat([torch.zeros([num_dx_ids, num_visit + num_demographics + num_vital_signs + num_dx_ids]),
                              torch.ones([num_dx_ids, num_proc_ids])], axis=1)
            row4 = torch.zeros([num_proc_ids, num_codes])
            guide = torch.cat([row0, row1, row2, row3, row4], axis=0)
            guide = guide + guide.t()
            guide = guide.to(device)
            guide = guide.unsqueeze(0)
            guide = guide.expand(batch_size, -1, -1)

            guide = (guide * masks.unsqueeze(-1) * masks.unsqueeze(1) + torch.eye(num_codes).to(device).unsqueeze(0))

        prior_guide = None
        if self.use_prior:
            prior_idx = priors['indices'].t()
            temp_idx = (prior_idx[:, 0] * 100000 + prior_idx[:, 1] * 1000 + prior_idx[:, 2])
            sorted_idx = torch.argsort(temp_idx)
            prior_idx = prior_idx[sorted_idx]

            prior_idx_shape = [batch_size, self.max_num_codes * 2, self.max_num_codes * 2]
            sparse_prior = torch.sparse_coo_tensor(prior_idx.t(), priors['values'], torch.Size(prior_idx_shape))
            prior_guide = sparse_prior.to_dense()

            visit_guide = torch.tensor(([self.prior_scalar] * self.max_num_codes + [
                0.0] * self.max_num_codes), dtype=torch.float, device=device)
            visit_guide = visit_guide.repeat(num_visit, 1)
            static_guide = torch.zeros(num_static, self.max_num_codes * 2, dtype=torch.float, device=device)

            prior_guide = torch.cat([visit_guide.unsqueeze(0).expand(batch_size, -1, -1),
                                     static_guide.unsqueeze(0).expand(batch_size, -1, -1), prior_guide], axis=1)
            zeros = torch.zeros(num_static + num_visit, num_visit, dtype=torch.float, device=device)
            visit_guide = torch.cat([zeros, visit_guide.transpose(0, 1)], axis=0)

            zeros = torch.zeros(num_static + num_visit, num_static, dtype=torch.float, device=device)
            static_guide = torch.cat([zeros, static_guide.transpose(0, 1)], axis=0)

            prior_guide = torch.cat([visit_guide.unsqueeze(0).expand(batch_size, -1, -1),
                                     static_guide.unsqueeze(0).expand(batch_size, -1, -1),
                                     prior_guide], axis=2)

            prior_guide = (
                    prior_guide * masks.unsqueeze(-1) * masks.unsqueeze(1) + self.prior_scalar * torch.eye(num_codes, device=device).unsqueeze(0))
            degrees = torch.sum(prior_guide, axis=2)
            prior_guide = prior_guide / degrees.unsqueeze(-1)
        return guide, prior_guide

    def get_loss(self, logits, labels, attentions):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.use_prior:
            kl_terms = []
            for i in range(1, self.num_stacks):
                log_p = torch.log(attentions[i - 1] + 1e-12)
                log_q = torch.log(attentions[i] + 1e-12)
                kl_term = attentions[i - 1] * (log_p - log_q)
                kl_term = torch.sum(kl_term, axis=-1)
                kl_term = torch.mean(kl_term)
                kl_terms.append(kl_term)
            reg_term = torch.mean(torch.tensor(kl_terms))
            loss += self.reg_coef * reg_term
        return loss

    def forward(self, data, priors_data, output_attentions=True):
        embedding_dict, mask_dict = self.embeddings(data)
        mask_dict['dx_ints'] = data['dx_masks']
        mask_dict['proc_ints'] = data['proc_masks']

        dx_visit_outputs, dx_dag_outputs, dx_code_attentions = self.dx_knowledge_encoder(
            data['dx_ints'], mask_dict['dx_ints'], output_attentions)
        proc_visit_outputs, proc_dag_outputs, proc_code_attentions = self.proc_knowledge_encoder(
            data['proc_ints'], mask_dict['proc_ints'], output_attentions)

        hidden_states = torch.cat([embedding_dict['visit'], embedding_dict['demographics_ints'],
                                   embedding_dict['vital_signs_ints'], dx_visit_outputs, proc_visit_outputs], dim=1)

        keys = ['visit', 'dx_ints', 'proc_ints', 'demographics_ints', 'vital_signs_ints']
        masks = torch.cat([mask_dict[key] for key in keys], axis=1)

        guide, prior_guide = self.create_matrix_vdp(data, masks, priors_data)

        all_hidden_states = () if self.output_hidden_states else None
        all_attentions = () if self.output_attentions else None
        extended_attention_mask = get_extended_attention_mask(masks)
        extended_guide_mask = get_extended_attention_mask(guide) if self.use_guide else None

        for i, layer_module in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, extended_attention_mask, extended_guide_mask, prior_guide, self.output_attentions)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        pooled_output = self.pooler(hidden_states)

        pooled_output = self.dropout(pooled_output)
        logits_expired = self.classifier(pooled_output)

        label_expired = data['expired']
        loss_fct = nn.CrossEntropyLoss()
        focalloss = FocalLoss(gamma=2, alpha=0.35, num_classes=2)
        loss_prediction = focalloss(logits_expired.to('cpu'), label_expired.to('cpu'))
        total_loss = loss_prediction

        return tuple(v for v in [total_loss, logits_expired, all_hidden_states, all_attentions] if v is not None)

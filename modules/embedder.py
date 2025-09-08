import torch
from torch import nn


class FeatureEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embeddings = {}
        self.feature_keys = args.feature_keys
        
        datasetName = args.dataset_name
        demographics_vocab_size = getattr(args, f"{datasetName}_vocab_sizes")['demographics_ints']
        vital_signs_vocab_size = getattr(args, f"{datasetName}_vocab_sizes")['vital_signs_ints']
        dx_vocab_size = getattr(args, f"{datasetName}_vocab_sizes")['dx_ints']
        proc_vocab_size = getattr(args, f"{datasetName}_vocab_sizes")['proc_ints']

        self.demographics_embeddings = nn.Embedding(demographics_vocab_size + 1, args.hidden_size, padding_idx=demographics_vocab_size)
        self.vital_signs_embeddings = nn.Embedding(vital_signs_vocab_size + 1, args.hidden_size, padding_idx=vital_signs_vocab_size)
        self.dx_embeddings = nn.Embedding(dx_vocab_size + 1, args.hidden_size, padding_idx=dx_vocab_size)
        self.proc_embeddings = nn.Embedding(proc_vocab_size + 1, args.hidden_size, padding_idx=proc_vocab_size)
        self.visit_embeddings = nn.Embedding(1, args.hidden_size)

        self.layernorm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, features):
        batch_size = features[self.feature_keys[0]].shape[0]

        embeddings = {}
        masks = {}
        embeddings['demographics_ints'] = self.demographics_embeddings(features['demographics_ints'])
        embeddings['vital_signs_ints'] = self.vital_signs_embeddings(features['vital_signs_ints'])
        embeddings['dx_ints'] = self.dx_embeddings(features['dx_ints'])
        proc_tensor = features['proc_ints'].int()
        embeddings['proc_ints'] = self.proc_embeddings(proc_tensor)

        device = features['dx_ints'].device
        embeddings['visit'] = self.visit_embeddings(torch.tensor([0]).to(device))
        embeddings['visit'] = embeddings['visit'].unsqueeze(0).expand(batch_size, -1, -1)
        masks['visit'] = torch.ones(batch_size, 1).to(device)
        masks['demographics_ints'] = torch.ones(batch_size, 9).to(device)
        masks['vital_signs_ints'] = torch.ones(batch_size, 70).to(device)

        return embeddings, masks

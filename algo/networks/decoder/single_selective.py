import torch
import torch.nn as nn


class GeneralAttentionScore(nn.Module):
    def __init__(self, query_size, key_size, attention_size):
        super().__init__()
        self._query_size = query_size
        self._key_size = key_size
        self._attention_size = attention_size
        self._w = nn.Linear(query_size + attention_size, 1)

    def forward(self, inputs):
        q, k = inputs
        q = q.unsqueeze(1).repeat(1, k.shape[1], 1)
        return self._w(torch.concat([q, k], dim=-1)).squeeze(-1)


class SingleSelectDecoder(nn.Module):
    def __init__(self, source_embedding_dim, target_embedding_dim, attention_size=64):
        super().__init__()
        self._attention_score = GeneralAttentionScore(
            source_embedding_dim, target_embedding_dim, attention_size
        )
        self._layers = nn.Linear(source_embedding_dim, 3)

    def forward(self, source_embeddings, target_embeddings):
        # attention_logits is uesd to choose character
        # while additional_logits is used to choose non-character
        attention_logits = self._attention_score((source_embeddings, target_embeddings))
        additional_logits = self._layers(source_embeddings)
        return torch.concat([attention_logits, additional_logits], dim=-1)

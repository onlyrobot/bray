import torch
import torch.nn as nn


class AttentionEntityEncoder(nn.Module):
    def __init__(
        self, entity_dim, embed_dim, num_heads, hidden_layer_sizes, main_entity_dim=None
    ):
        super().__init__()
        self._layers = nn.Sequential()
        for layer_id, layer_size in enumerate(hidden_layer_sizes):
            self._layers.add_module(
                "linear_{}".format(layer_id), nn.Linear(entity_dim, layer_size)
            )
            self._layers.add_module("relu_{}".format(layer_id), nn.ReLU())
            entity_dim = layer_size

        # entity_dim equals to last layer size
        self._dense = (
            nn.Linear(main_entity_dim, embed_dim)
            if main_entity_dim is not None
            else None
        )
        self._attention = nn.MultiheadAttention(
            embed_dim, num_heads, kdim=entity_dim, vdim=entity_dim, batch_first=True
        )

        self.output_dim = embed_dim

    def forward(self, input, main_embedding=None):
        """
        Main_embedding is not None iff this is a cross attention encoder.
        [Applicable scenarios] Cross attention between entity and character.
            main_embedding: [batch_size, embedding_size]
            entity_embeddings: [batch_size, num_entity, embedding_size]
        """
        entity_embeddings = self._layers(input)
        if main_embedding is None:
            attn_outputs, attn_weights = self._attention(
                entity_embeddings,
                entity_embeddings,
                entity_embeddings,
            )
        else:
            main_embedding = self._dense(main_embedding).unsqueeze(1)
            attn_outputs, attn_weights = self._attention(
                main_embedding,
                entity_embeddings,
                entity_embeddings,
            )
        return attn_outputs, entity_embeddings


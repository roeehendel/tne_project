from models.modules.np_contextual_embedding.base_np_contextual_embedder import BaseNPContextualEmbedder


class PassthroughNPContextualEmbedder(BaseNPContextualEmbedder):
    @property
    def output_size(self):
        return self.input_size

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        np_embeddings = intermediate_outputs['np_embedder']['embeddings']
        return dict(embeddings=np_embeddings)

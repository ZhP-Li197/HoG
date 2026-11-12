import numpy as np
from sentence_transformers import SentenceTransformer

from graph import norm


class Encoders:
    def __init__(self, entity_model, statement_model):
        self.entity_model = SentenceTransformer(entity_model)
        self.statement_model = self.entity_model if entity_model == statement_model else SentenceTransformer(statement_model)
        self.entity_index = None
        self.entity_keys = None

    def link(self, graph, mention, k=3):
        exact = graph.entity_candidates(mention)
        if exact:
            return exact[:k]
        if self.entity_index is None:
            self.entity_keys = list(graph.entities)
            labels = [graph.entities[key] for key in self.entity_keys]
            self.entity_index = self.entity_model.encode(labels, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
        vector = self.entity_model.encode([mention], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.entity_index @ vector
        indices = np.argsort(-scores)[:k]
        return [self.entity_keys[int(index)] for index in indices]

    def top_statements(self, graph, question, statement_ids, k=10):
        ids = sorted(statement_ids)
        if len(ids) <= k:
            return [graph.statements[sid] for sid in ids]
        texts = [graph.serialized(graph.statements[sid]) for sid in ids]
        vectors = self.statement_model.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
        query = self.statement_model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = vectors @ query
        indices = np.argsort(-scores)[:k]
        return [graph.statements[ids[int(index)]] for index in indices]

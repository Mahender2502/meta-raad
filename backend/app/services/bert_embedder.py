"""
Embeds text with bert-base-uncased the way NLP-ADBench produced its precomputed
vectors, so a typed sentence lands in the same space as the stored train vectors.

Verified recipe: CLS token of the last hidden layer, input truncated to 512 tokens
(cosine 1.0000 against stored rows). Not a sentence-transformer: raw BERT CLS.
"""

from functools import lru_cache

import numpy as np

from app.core.config import settings


class BertEmbedder:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        import torch  # local imports keep the module importable without the ML stack
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name or settings.bert_model
        self.device = device or settings.bert_device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def embed(self, text: str) -> np.ndarray:
        """Raw (unnormalized) 768-d CLS vector for one text."""
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._torch.no_grad():
            hidden = self._model(**enc).last_hidden_state
        return hidden[0, 0].cpu().numpy().astype(np.float32)


@lru_cache(maxsize=1)
def get_bert_embedder() -> BertEmbedder:
    """Process-wide singleton so the model is loaded only once."""
    return BertEmbedder()

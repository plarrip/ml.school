import time

import litellm
import tiktoken
from langchain_core.embeddings import Embeddings

# Free-tier Gemini API keys enforce a 30,000 input-tokens-per-minute quota, so we
# group documents into batches that stay comfortably under that limit and pause
# between batches to let the quota window reset.
TOKENS_PER_MINUTE_LIMIT = 25_000
BATCH_DELAY_SECONDS = 65

_encoding = tiktoken.get_encoding("cl100k_base")


class CustomEmbeddingModel(Embeddings):
    """Custom text embedding implementation model.

    This is the implementation of the `Embeddings` interface to map text to vectors.
    This implementation uses LiteLLM to allow flexible model selection to generate
    embeddings.
    """

    def __init__(self, model: str) -> None:
        """Initialize the embedding model."""
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed the supplied list of documents."""
        embeddings: list[list[float]] = []
        batch: list[str] = []
        batch_tokens = 0

        for text in texts:
            text_tokens = len(_encoding.encode(text))

            if batch and batch_tokens + text_tokens > TOKENS_PER_MINUTE_LIMIT:
                embeddings.extend(self._embed_batch(batch))
                time.sleep(BATCH_DELAY_SECONDS)
                batch, batch_tokens = [], 0

            batch.append(text)
            batch_tokens += text_tokens

        if batch:
            embeddings.extend(self._embed_batch(batch))

        return embeddings

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed a single batch of documents that fits within the quota."""
        response = litellm.embedding(model=self.model, input=batch, num_retries=5)
        return [d["embedding"] for d in response["data"]]

    def embed_query(self, text: str) -> list[float]:
        """Embed the supplied query text."""
        return self.embed_documents([text])[0]

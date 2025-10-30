"""Stub embedder for lightweight CI testing.

Used when EMBEDDINGS_BACKEND=stub to avoid downloading SentenceTransformer models.
Returns deterministic 384-dimensional vectors based on text hash.
"""

import numpy as np
from typing import List, Union


class StubEmbedder:
    """Lightweight stub embedder for CI testing.

    Generates deterministic embeddings by hashing input text.
    Useful for tests that need embeddings but don't require semantic meaning.
    """

    def __init__(self, model_name: str = "stub", max_seq_length: int = 512):
        """Initialize stub embedder.

        Args:
            model_name: Ignored (for API compatibility)
            max_seq_length: Ignored (for API compatibility)
        """
        self.model_name = "stub"
        self.max_seq_length = max_seq_length
        self.embedding_dim = 384  # Match all-MiniLM-L6-v2 output dim

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """Generate deterministic embeddings for given sentences.

        Args:
            sentences: Single sentence or list of sentences
            convert_to_numpy: If True, return numpy array (else list)
            normalize_embeddings: If True, L2-normalize embeddings

        Returns:
            Embeddings as numpy array or list of arrays (clamped to [-5, 5])
        """
        # Handle single string
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for sentence in sentences:
            # Create deterministic hash-based embedding
            # Hash the sentence to get a seed
            seed = hash(sentence) % (2 ** 31)
            rng = np.random.default_rng(seed)

            # Generate 384-dim embedding
            embedding = rng.standard_normal(self.embedding_dim).astype("float32")

            # Clamp values to prevent extreme values in stub mode
            # Standard normal can produce values beyond [-5, 5], clamp them
            embedding = np.clip(embedding, -5.0, 5.0).astype("float32")

            # Normalize if requested
            if normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            embeddings.append(embedding)

        # Return as numpy array if convert_to_numpy, else list
        if convert_to_numpy:
            return np.array(embeddings)
        return embeddings

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Make embedder callable for compatibility."""
        return self.encode(*args, **kwargs)


def get_stub_embedder() -> StubEmbedder:
    """Get or create stub embedder instance."""
    return StubEmbedder()

"""
intentguard.embedder
====================
Thin wrapper around the ``sentence-transformers`` library for loading and
running a pre-trained sentence embedding model.

Model used by default: ``all-MiniLM-L6-v2``
    - 384-dimensional output vectors
    - ~22 M parameters; fast CPU inference
    - Strong performance on semantic similarity benchmarks

Design notes
------------
* The model is loaded **lazily** on first call so that importing this module
  does not trigger a potentially slow network/disk read at import time.
* A module-level singleton (_model) is used so the model is loaded at most
  once per process.
* Both encode() and encode_batch() normalise their output to unit vectors
  (L2 norm = 1) so that a dot product equals cosine similarity.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Module-level singleton – populated lazily by _get_model()
# ---------------------------------------------------------------------------
_MODEL_NAME: str = "all-MiniLM-L6-v2"
_EMBEDDING_DIM: int = 384
_model = None  # SentenceTransformer | None


def _get_model():
    """Return the cached SentenceTransformer model, loading it if necessary.

    The first call downloads / reads the model from disk and caches it in
    the module-level ``_model`` variable.  Subsequent calls return the cached
    instance instantly.

    Returns
    -------
    sentence_transformers.SentenceTransformer
        The loaded model ready for inference.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.
    OSError
        If the model files cannot be found or downloaded.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required by intentguard. "
                "Install it with:  pip install sentence-transformers"
            ) from exc

        _model = SentenceTransformer(_MODEL_NAME)

    return _model


def encode(text: str) -> np.ndarray:
    """Encode a single string into a normalised embedding vector.

    Parameters
    ----------
    text : str
        The input text to embed (e.g. a user intent sentence or an API-call
        description string built from its name + argument values).
        Must be a non-empty string.

    Returns
    -------
    np.ndarray
        Shape ``(384,)``, dtype ``float32``.  The vector is L2-normalised so
        that ``np.dot(a, b) == cosine_similarity(a, b)`` for any two encoded
        texts *a* and *b*.

    Raises
    ------
    ValueError
        If *text* is ``None``, not a string, or an empty / whitespace-only string.

    Examples
    --------
    >>> vec = encode("Book a flight to Paris")
    >>> vec.shape
    (384,)
    >>> import numpy as np; abs(np.linalg.norm(vec) - 1.0) < 1e-5
    True
    """
    if text is None:
        raise ValueError("encode() received None; expected a non-empty string.")
    if not isinstance(text, str):
        raise ValueError(
            f"encode() expected a str, got {type(text).__name__!r}."
        )
    if not text.strip():
        raise ValueError(
            "encode() received an empty or whitespace-only string. "
            "Provide a non-empty text to embed."
        )

    model = _get_model()
    # encode() with convert_to_numpy=True (default) + normalize_embeddings=True
    # returns a 2-D array of shape (1, 384); [0] extracts the single row.
    vector: np.ndarray = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # Guarantee dtype and contiguity regardless of model/platform defaults.
    return np.ascontiguousarray(vector, dtype=np.float32)


def encode_batch(texts: list[str]) -> np.ndarray:
    """Encode multiple strings in a single forward pass.

    Batching is significantly faster than calling ``encode()`` in a loop
    because it allows the underlying model to make full use of vectorised
    matrix operations.

    Parameters
    ----------
    texts : list[str]
        A non-empty list of non-empty strings.  Order is preserved.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), 384)``, dtype ``float32``.  Each row is the
        L2-normalised embedding of the corresponding input string, identical
        in semantics to calling ``encode()`` on each string individually.

    Raises
    ------
    ValueError
        If *texts* is ``None``, not a list, empty, or contains any
        ``None`` / empty / whitespace-only element.

    Examples
    --------
    >>> vecs = encode_batch(["hello", "world"])
    >>> vecs.shape
    (2, 384)
    """
    if texts is None:
        raise ValueError(
            "encode_batch() received None; expected a non-empty list of strings."
        )
    if not isinstance(texts, list):
        raise ValueError(
            f"encode_batch() expected a list, got {type(texts).__name__!r}."
        )
    if len(texts) == 0:
        raise ValueError(
            "encode_batch() received an empty list. "
            "Provide at least one string to embed."
        )

    # Validate every element before touching the model so errors are caught
    # early with an actionable message rather than a confusing model-internal
    # crash.
    for i, item in enumerate(texts):
        if item is None:
            raise ValueError(
                f"encode_batch() received None at index {i}; "
                "all list elements must be non-empty strings."
            )
        if not isinstance(item, str):
            raise ValueError(
                f"encode_batch() received {type(item).__name__!r} at index {i}; "
                "all list elements must be strings."
            )
        if not item.strip():
            raise ValueError(
                f"encode_batch() received an empty or whitespace-only string "
                f"at index {i}. All elements must be non-empty."
            )

    model = _get_model()
    # Single batched forward pass — much faster than N individual encode() calls.
    matrix: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Guarantee dtype and C-contiguous layout.
    return np.ascontiguousarray(matrix, dtype=np.float32)

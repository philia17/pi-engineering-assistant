"""Custom retrieval strategies layered on top of the dense pipeline."""

from rag.retrievers.rerank import rerank_chunks

__all__ = ["rerank_chunks"]

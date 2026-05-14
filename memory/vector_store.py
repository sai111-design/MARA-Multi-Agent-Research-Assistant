"""Qdrant vector store integration for MARA.

Provides helper functions to initialise a Qdrant collection, upsert document
embeddings produced by sentence-transformers, and run similarity searches that
supply long-term memory to the agent pipeline.
"""

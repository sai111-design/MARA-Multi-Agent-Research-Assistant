"""LangGraph orchestration for the MARA pipeline.

Constructs the StateGraph, registers agent nodes (planner, researcher, critic,
synthesizer), defines edges and conditional routing, and compiles the runnable graph.
"""

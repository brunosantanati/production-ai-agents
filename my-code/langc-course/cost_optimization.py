"""
Cost Optimization Patterns
Reducing LLM costs in production
"""

import hashlib
import json
from typing import Optional, Callable
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

# === Model Routing ===


class ModelRouter:
    """Route queries to appropriate model based on complexity."""

    def __init__(self):
        self.cheap_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.expensive_model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.classifier = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def classify_complexity(self, query: str) -> str:
        """Classify query complexity."""

        prompt = ChatPromptTemplate.from_template(
            """
Classify this query's complexity as 'simple' or 'complex'.

Simple: Basic facts, short answers, simple calculations
Complex: Analysis, reasoning, creative tasks, multi-step problems

Query: {query}

Respond with only: simple or complex
"""
        )

        response = self.classifier.invoke(prompt.format(query=query))
        return response.content.strip().lower()

    @traceable(name="routed_query")
    def invoke(self, query: str) -> tuple[str, str, float]:
        """
        Route and invoke query.
        Returns: (response, model_used, estimated_cost)
        """
        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            model_name = "gpt-4o-mini"
            cost_per_1k = 0.00015  # Input cost
        else:
            model = self.expensive_model
            model_name = "gpt-4o"
            cost_per_1k = 0.0025  # Input cost

        response = model.invoke(query)

        # Estimate cost (rough)
        tokens = len(query.split()) * 1.3  # Rough token estimate
        estimated_cost = (tokens / 1000) * cost_per_1k

        return response.content, model_name, estimated_cost


def demo_model_routing():
    """Demonstrate model routing."""

    router = ModelRouter()

    queries = [
        "What is 2 + 2?",  # Simple
        "Analyze the economic implications of AI on the job market.",  # Complex
        "What color is the sky?",  # Simple
    ]

    print("Model Routing Demo:\n")

    total_cost = 0
    for query in queries:
        result, model, cost = router.invoke(query)
        total_cost += cost
        print(f"Query: {query[:50]}...")
        print(f"  Model: {model}")
        print(f"  Est. Cost: ${cost:.6f}")
        print(f"  Response: {result[:50]}...")

    print(f"\nTotal Estimated Cost: ${total_cost:.6f}")


# === Semantic Caching ===


class SemanticCache:
    """Cache responses with semantic similarity matching."""

    def __init__(self, similarity_threshold: float = 0.9):
        self.cache = {}
        self.threshold = similarity_threshold
        self.embedder = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _hash_query(self, query: str) -> str:
        """Create hash of normalized query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Get cached response if similar query exists."""
        query_hash = self._hash_query(query)

        # Exact match
        if query_hash in self.cache:
            return self.cache[query_hash]["response"]

        # Could add embedding-based similarity here
        # For demo, just use exact match

        return None

    def set(self, query: str, response: str):
        """Cache a response."""
        query_hash = self._hash_query(query)
        self.cache[query_hash] = {"query": query, "response": response}

    def stats(self) -> dict:
        return {"cached_queries": len(self.cache)}


class CachedLLM:
    """LLM wrapper with caching."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.cache = SemanticCache()
        self.cache_hits = 0
        self.cache_misses = 0

    @traceable(name="cached_invoke")
    def invoke(self, query: str) -> tuple[str, bool]:
        """
        Invoke with caching.
        Returns: (response, from_cache)
        """
        # Check cache
        cached = self.cache.get(query)
        if cached:
            self.cache_hits += 1
            return cached, True

        # Call LLM
        self.cache_misses += 1
        response = self.llm.invoke(query)
        result = response.content

        # Cache result
        self.cache.set(query, result)

        return result, False

    def get_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


def demo_caching():
    """Demonstrate caching."""

    llm = CachedLLM()

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Cache hit
        "What is python?",  # Cache hit (normalized)
        "What is Rust?",
    ]

    print("\nCaching Demo:\n")

    for query in queries:
        result, from_cache = llm.invoke(query)
        source = "CACHE" if from_cache else "LLM"
        print(f"[{source}] {query} -> {result[:30]}...")

    print(f"\nStats: {llm.get_stats()}")


if __name__ == "__main__":
    # demo_model_routing()
    demo_caching()

    # Production version would:
# 1. Embed the query into a vector
# 2. Search the cache by vector similarity
# 3. Return if similarity > threshold (e.g., 0.95)

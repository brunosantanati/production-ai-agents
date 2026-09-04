"""
Error Handling and Reliability Patterns
Building robust LangGraph applications
"""

import time
import random
from typing import Literal, Optional, Callable
from functools import wraps
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
import operator
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()


# === Retry Decorator ===


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        # Add jitter
                        delay = delay * (0.5 + random.random())
                        print(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


@with_retry(max_retries=3, base_delay=1.0)
def unreliable_api_call(query: str) -> str:
    """Simulates an unreliable API."""
    if random.random() < 0.5:
        raise ConnectionError("Simulated API failure")
    return f"Success: {query}"


def demo_retry_pattern():
    """Demonstrate retry with exponential backoff."""

    print("Retry Pattern Demo:\n")

    for i in range(3):
        try:
            result = unreliable_api_call(f"Query {i}")
            print(f"✅ {result}")
        except Exception as e:
            print(f"❌ Failed after retries: {e}")


# === Circuit Breaker ===


class CircuitBreaker:
    """Circuit breaker pattern for failing services."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        # Check if circuit should move from open to half-open
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            # Success - reset on half-open
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = "open"

            raise e


def demo_circuit_breaker():
    """Demonstrate circuit breaker pattern."""

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    def flaky_service():
        if random.random() < 0.7:
            raise Exception("Service error")
        return "OK"

    print("\nCircuit Breaker Demo:\n")

    for i in range(15):
        try:
            result = breaker.call(flaky_service)
            print(f"Attempt {i+1}: ✅ {result} (state: {breaker.state})")
        except Exception as e:
            print(f"Attempt {i+1}: ❌ {e} (state: {breaker.state})")

        # After attempt 7, wait long enough for recovery
        if i == 6:
            print("  ⏳ Waiting 6 seconds for recovery timeout...")
            time.sleep(6)
        else:
            time.sleep(0.5)


# === Model Fallback Chain ===


class FallbackChain:
    """Try multiple models in order until one succeeds."""

    def __init__(self):
        self.models = [
            ("gpt-4o-mini", ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=10)),
            ("gpt-4o", ChatOpenAI(model="gpt-4o", temperature=0, timeout=10)),
            (
                "claude-sonnet",
                ChatAnthropic(
                    model="claude-sonnet-4-5-20250929", temperature=0, timeout=10
                ),
            ),
        ]
        self.cache = {}

    @traceable(name="fallback_invoke")
    def invoke(self, query: str, use_cache: bool = True) -> tuple[str, str]:
        """
        Invoke with fallbacks.
        Returns: (response, model_used)
        """

        # Check cache first
        if use_cache and query in self.cache:
            return self.cache[query], "cache"

        errors = []

        for model_name, model in self.models:
            try:
                response = model.invoke(query)
                result = response.content

                # Cache successful response
                self.cache[query] = result

                return result, model_name

            except Exception as e:
                errors.append(f"{model_name}: {str(e)}")
                continue

        # All models failed
        raise Exception(f"All models failed: {errors}")


def demo_fallback_chain():
    """Demonstrate fallback chain."""

    chain = FallbackChain()

    print("\nFallback Chain Demo:\n")

    queries = [
        "What is 2 + 2?",
        "What is Python?",
        "What is 2 + 2?",  # Should hit cache
    ]

    for query in queries:
        try:
            result, model = chain.invoke(query)
            print(f"Query: {query}")
            print(f"  Model: {model}")
            print(f"  Response: {result[:50]}...")
        except Exception as e:
            print(f"Query: {query}")
            print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    # Example usage of the unreliable API call with retry logic
    # try:
    #     result = unreliable_api_call("Hello, World!")
    #     print(result)
    # except Exception as e:
    #     print(f"API call failed after retries: {e}")

    # Run the retry pattern demonstration
    # demo_retry_pattern()

    # Run the circuit breaker demonstration
    # demo_circuit_breaker()

    # Run the fallback chain demonstration
    demo_fallback_chain()
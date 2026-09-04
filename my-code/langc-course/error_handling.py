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


if __name__ == "__main__":
    # Example usage of the unreliable API call with retry logic
    # try:
    #     result = unreliable_api_call("Hello, World!")
    #     print(result)
    # except Exception as e:
    #     print(f"API call failed after retries: {e}")

    # Run the retry pattern demonstration
    demo_retry_pattern()
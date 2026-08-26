"""
Testing & Evaluation Patterns
Building reliable LLM applications
"""

import pytest
from unittest.mock import Mock, patch
from typing import Callable
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langsmith import traceable, Client
from dotenv import load_dotenv

load_dotenv()


# === Unit Testing with Mocks ===
class QAChain:
    """Simple Q&A chain for testing."""

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            "Answer this question: {question}"
        )

    def ask(self, question: str) -> str:
        prompt_value = self.prompt.invoke({"question": question})
        response = self.llm.invoke(prompt_value)
        return response.content


def test_qa_chain_with_mock():
    """Test QA chain with mocked LLM."""

    # Create mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(content="Paris")

    # Test with mock
    chain = QAChain(llm=mock_llm)
    result = chain.ask("What is the capital of France?")

    assert result == "Paris"
    mock_llm.invoke.assert_called_once()


def test_qa_chain_handles_empty_response():
    """Test chain handles empty responses."""

    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(content="")

    chain = QAChain(llm=mock_llm)
    result = chain.ask("Empty question")

    assert result == ""


# === Integration Testing with Real LLM ===
class IntegrationTestSuite:
    """Integration tests with real LLM calls."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @traceable(name="integration_test")
    def test_basic_qa(self) -> dict:
        """Test basic question answering."""

        test_cases = [
            {
                "question": "What is 2 + 2?",
                "expected_contains": ["4", "four"],
            },
            {
                "question": "What color is the sky on a clear day?",
                "expected_contains": ["blue"],
            },
        ]

        results = []
        for case in test_cases:
            response = self.llm.invoke(case["question"])
            content = response.content.lower()

            passed = any(exp.lower() in content for exp in case["expected_contains"])

            # "The answer is 4" or "2 + 2 equals four" or "That would be 4."

            results.append(
                {
                    "question": case["question"],
                    "response": response.content,
                    "passed": passed,
                }
            )

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "results": results,
        }


def demo_integration_tests():
    """Run integration tests."""

    suite = IntegrationTestSuite()

    print("Integration Test Results:\n")

    results = suite.test_basic_qa()

    print(f"Passed: {results['passed']}/{results['total']}")

    for r in results["results"]:
        status = "✅" if r["passed"] else "❌"
        print(f"{status} {r['question']}")
        print(f"   Response: {r['response'][:50]}...")


if __name__ == "__main__":
    # test_qa_chain_with_mock()
    # print("All tests passed!")
    # test_qa_chain_handles_empty_response()
    # print("All tests passed!")
    demo_integration_tests()
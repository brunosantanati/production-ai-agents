"""
Agent Communication Patterns in LangGraph
Shared state, message passing, and blackboard pattern
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
import operator
import json
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Pattern 1: Message Passing
# Agents communicate through a shared message list
# ============================================================


class MessagePassingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_phase: str


def create_message_passing_pipeline():
    """Agents communicate by appending messages that others can read."""

    def researcher(state: MessagePassingState) -> dict:
        """Researches the topic and posts findings as a message."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a researcher. Read the user's question, "
                        "research it, and post your findings. Keep it to 2-3 sentences."
                    )
                ),
                *state["messages"],
            ]
        )
        return {
            "messages": [
                AIMessage(
                    content=f"[RESEARCHER]: {response.content}", name="researcher"
                )
            ],
            "current_phase": "fact_checker",
        }

    def fact_checker(state: MessagePassingState) -> dict:
        """Reads the researcher's message and validates the claims."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a fact-checker. Read the researcher's findings "
                        "in the conversation and validate or challenge them. "
                        "Keep it to 2-3 sentences."
                    )
                ),
                *state["messages"],
            ]
        )
        return {
            "messages": [
                AIMessage(
                    content=f"[FACT-CHECKER]: {response.content}", name="fact_checker"
                )
            ],
            "current_phase": "summarizer",
        }

    def summarizer(state: MessagePassingState) -> dict:
        """Reads all previous messages and creates a final summary."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a summarizer. Read the researcher's findings and "
                        "the fact-checker's review. Produce a final, accurate summary. "
                        "Keep it to 2-3 sentences."
                    )
                ),
                *state["messages"],
            ]
        )
        return {
            "messages": [
                AIMessage(content=f"[SUMMARY]: {response.content}", name="summarizer")
            ],
            "current_phase": "done",
        }

    graph = StateGraph(MessagePassingState)

    graph.add_node("researcher", researcher)
    graph.add_node("fact_checker", fact_checker)
    graph.add_node("summarizer", summarizer)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "fact_checker")
    graph.add_edge("fact_checker", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


def demo_message_passing():
    """Demo message passing between agents."""
    agent = create_message_passing_pipeline()

    print("Message Passing Demo:\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="What are the main benefits of renewable energy?")
            ],
            "current_phase": "researcher",
        }
    )

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"{msg.content}\n")


# ============================================================
# Pattern 2: Shared State (Typed Fields)
# Agents communicate through structured state fields
# ============================================================
class SharedFieldsState(TypedDict):
    query: str
    # Each agent writes to its own field — others can read it
    raw_data: Annotated[list[dict], operator.add]
    analysis: str
    recommendations: list[str]
    confidence_score: float


def create_shared_fields_pipeline():
    """Agents communicate through typed state fields, not messages."""

    def data_collector(state: SharedFieldsState) -> dict:
        """Collects data and writes to the raw_data field."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a data collector. Given the query, produce 3 data points "
                        "as a JSON array of objects with 'source' and 'finding' keys. "
                        "Return ONLY the JSON array, no markdown."
                    )
                ),
                HumanMessage(content=state["query"]),
            ]
        )

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            data = [{"source": "llm", "finding": response.content}]

        return {"raw_data": data}

    def analyst(state: SharedFieldsState) -> dict:
        """Reads raw_data field, writes analysis and confidence."""
        data_summary = json.dumps(state["raw_data"], indent=2)

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a data analyst. Analyze the collected data and provide: "
                        "1) A brief analysis (2-3 sentences), and "
                        "2) A confidence score from 0.0 to 1.0. "
                        "Format: ANALYSIS: <text>\nCONFIDENCE: <number>"
                    )
                ),
                HumanMessage(
                    content=f"Query: {state['query']}\n\nData:\n{data_summary}"
                ),
            ]
        )

        content = response.content
        analysis = content
        confidence = 0.7  # default

        if "CONFIDENCE:" in content:
            parts = content.split("CONFIDENCE:")
            analysis = parts[0].replace("ANALYSIS:", "").strip()
            try:
                confidence = float(parts[1].strip())
            except ValueError:
                confidence = 0.7

        return {"analysis": analysis, "confidence_score": confidence}

    def advisor(state: SharedFieldsState) -> dict:
        """Reads analysis + confidence, writes recommendations."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strategic advisor. Based on the analysis and "
                        "confidence score, provide 3 actionable recommendations. "
                        "Return them as a JSON array of strings. "
                        "Return ONLY the JSON array, no markdown."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Query: {state['query']}\n"
                        f"Analysis: {state['analysis']}\n"
                        f"Confidence: {state['confidence_score']}"
                    )
                ),
            ]
        )

        try:
            recs = json.loads(response.content)
        except json.JSONDecodeError:
            recs = [response.content]

        return {"recommendations": recs}

    graph = StateGraph(SharedFieldsState)

    graph.add_node("data_collector", data_collector)
    graph.add_node("analyst", analyst)
    graph.add_node("advisor", advisor)

    graph.add_edge(START, "data_collector")
    graph.add_edge("data_collector", "analyst")
    graph.add_edge("analyst", "advisor")
    graph.add_edge("advisor", END)

    return graph.compile()


def demo_shared_state():
    """Demo shared state fields between agents."""
    agent = create_shared_fields_pipeline()

    print("Shared State Demo:\n")

    result = agent.invoke(
        {
            "query": "Should a small business invest in AI automation in 2026?",
            "raw_data": [],
            "analysis": "",
            "recommendations": [],
            "confidence_score": 0.0,
        }
    )

    print(f"Data collected: {len(result['raw_data'])} points")
    for d in result["raw_data"]:
        print(f"  - [{d.get('source', 'N/A')}] {d.get('finding', 'N/A')[:80]}...")

    print(f"\nAnalysis: {result['analysis'][:200]}...")
    print(f"Confidence: {result['confidence_score']}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(result["recommendations"], 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    # demo_message_passing()
    demo_shared_state()

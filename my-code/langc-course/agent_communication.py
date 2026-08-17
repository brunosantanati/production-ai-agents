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


if __name__ == "__main__":
    demo_message_passing()

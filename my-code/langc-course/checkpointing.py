"""
Checkpointing and Persistence in LangGraph
Save and resume agent state
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import tempfile
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def demo_memory_saver():
    """In-memory checkpointing for development."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)

    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    saver = MemorySaver()
    app = graph.compile(checkpointer=saver)

    # Configuration with thread_id
    config = {"configurable": {"thread_id": "user-123"}}

    print("Memory Saver Demo (Multi-turn conversation):\n")

    # Turn 1
    result = app.invoke(
        {"messages": [HumanMessage(content="My name is Paulo")]}, config
    )
    print(f"Turn 1 - AI: {result['messages'][-1].content}")

    # Turn 2 - Conversation continues
    result = app.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
    print(f"Turn 2 - AI: {result['messages'][-1].content}")

    # Check full history
    state = app.get_state(config)
    print(f"\nTotal messages in state: {len(state.values['messages'])}")


if __name__ == "__main__":
    demo_memory_saver()

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


def demo_sqlite_persistence():
    """SQLite persistence for durable storage."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"\nSQLite Persistence Demo:")
    print(f"Database: {db_path}\n")

    # First session
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content="Remember: The secret code is ALPHA-7")
                ]
            },
            config,
        )
        print(f"Session 1 - Stored secret code")

        # PostgresSaver with a real database!
        # Simulate app restart - new session
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {"messages": [HumanMessage(content="What was the secret code?")]}, config
        )
        print(f"Session 2 - AI: {result['messages'][-1].content}")


if __name__ == "__main__":
    # demo_memory_saver()
    demo_sqlite_persistence()

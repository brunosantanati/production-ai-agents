"""
LangGraph Core Concepts
StateGraph, nodes, edges, and basic patterns
"""

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
from dotenv import load_dotenv

load_dotenv()


# Basic state
class SimpleState(TypedDict):
    input: str
    output: str
    step: int


def demo_simple_graph():
    # define node functions
    def process(state: SimpleState) -> dict:
        # simple processing logic, for demo purposes
        return {"output": state["input"].upper(), "step": state["step"] + 1}

    # create graph
    graph = StateGraph(SimpleState)

    # add nodes
    graph.add_node("process", process)
    # add edges
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    # execute graph/ compile
    app = graph.compile()

    # # visualize the graph
    # print("\n--- Mermaid Graph ---")
    # print(app.get_graph().draw_mermaid())

    # # save as PNG
    # png_bytes = app.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_bytes)
    # print("\nGraph saved to graph.png")

    # run app
    result = app.invoke({"input": "hello", "output": "", "step": 0})

    print("simple graph result:", result)
    print(
        f" Input: {result['input']}, Output: {result['output']}, Step: {result['step']}"
    )

# === State with Reducers ===


class AccumulatingState(TypedDict):
    messages: Annotated[list[str], operator.add]  # lists concatenate when merged
    count: Annotated[int, operator.add]  # counts sum when merged


def demo_accumulating_state():
    def step_one(state: AccumulatingState) -> dict:
        return {"messages": ["Step 1 executed"], "count": 1}

    def step_two(state: AccumulatingState) -> dict:
        return {"messages": ["Step 2 executed"], "count": 1}

    graph = StateGraph(AccumulatingState)

    print("\nGraph saved to graph_2.png")
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)

    app = graph.compile()

    # # visualize the graph
    print("\n--- Mermaid Graph ---")
    print(app.get_graph().draw_mermaid())

    # save as PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph_2.png", "wb") as f:
        f.write(png_bytes)

    result = app.invoke({"messages": ["Initial message"], "count": 0})

    print("\nAccumulating State Result:")
    print(f"  Messages: {result['messages']}")
    print(f"  Count: {result['count']}")

# === Message State (Common Pattern) ===

from langgraph.graph import add_messages


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def demo_message_state():
    llm = init_chat_model("gpt-4o-mini", temperature=0)

    def chat_node(state: MessageState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessageState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    app = graph.compile()

    result = app.invoke({"messages": [HumanMessage(content="Say Hello in Tagalog")]})

    print("\nMessage State Result:")
    for msg in result["messages"]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content}")

if __name__ == "__main__":
    # demo_simple_graph()
    # demo_accumulating_state()
    demo_message_state()

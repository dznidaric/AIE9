"""An agent graph with a post-response fact-verification loop.

After the agent responds, a secondary node evaluates whether the response
contains factually accurate and well-supported claims.
If verified, end; otherwise, continue the loop or terminate after a safe limit.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from app.state import MessagesState
from app.models import get_chat_model
from app.tools import get_tool_belt


class FactCheckResult(BaseModel):
    is_factual: bool = Field(
        description="Whether the response contains only factually accurate, well-supported claims"
    )
    reasoning: str = Field(
        description="Brief explanation of why the response is or isn't factually sound"
    )


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_fact_check(state: MessagesState):
    """Decide whether to execute tools or run the fact-check evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "fact_check"


_fact_check_prompt = ChatPromptTemplate.from_template(
    "You are a rigorous fact-checker. Given an initial query and the agent's "
    "final response, determine whether the response is factually accurate and "
    "well-supported. Flag any unsupported claims, hallucinations, or inaccuracies.\n\n"
    "Initial Query:\n{initial_query}\n\n"
    "Agent Response:\n{agent_response}"
)


def fact_check_node(state: MessagesState) -> dict:
    """Evaluate factual accuracy of the latest response relative to the initial query."""
    # Guard against infinite loops – stop after 10 messages
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="FACT_CHECK:END")]}

    initial_query = state["messages"][0]
    agent_response = state["messages"][-1]

    structured_model = get_chat_model(model_name="gpt-4.1-mini").with_structured_output(
        FactCheckResult
    )
    result = (_fact_check_prompt | structured_model).invoke(
        {
            "initial_query": initial_query.content,
            "agent_response": agent_response.content,
        }
    )

    if result.is_factual:
        return {"messages": [AIMessage(content="FACT_CHECK:PASS")]}
    # Feed the reasoning back so the agent can self-correct on the next loop
    return {
        "messages": [
            AIMessage(
                content=f"FACT_CHECK:FAIL – {result.reasoning}. "
                "Please revise your previous response to fix inaccuracies."
            )
        ]
    }


def fact_check_decision(state: MessagesState):
    """Route based on the fact-check verdict: pass → end, fail → retry, limit → end."""
    last = state["messages"][-1]
    text = getattr(last, "content", "")

    if "FACT_CHECK:END" in text:
        return END
    if "FACT_CHECK:PASS" in text:
        return "end"
    # FACT_CHECK:FAIL – loop back to agent for self-correction
    return "continue"


def build_graph():
    """Build an agent graph with an auxiliary fact-check evaluation loop."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())

    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("fact_check", fact_check_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_fact_check,
        {"action": "action", "fact_check": "fact_check"},
    )
    graph.add_conditional_edges(
        "fact_check",
        fact_check_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


# Export compiled graph for LangGraph
graph = build_graph().compile()

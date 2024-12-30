from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from .graph_state import GraphState
from .nodes import extract_context, handle_comparison, handle_stock_specific, handle_technical
from .edges import classify_question

memory = MemorySaver()

workflow = StateGraph(GraphState)


workflow.add_node("extract_context", extract_context)
workflow.add_node("handle_comparison", handle_comparison)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_stock_specific", handle_stock_specific)

workflow.add_edge(
  START, "extract_context"
)

workflow.add_conditional_edges(
  "extract_context",
  classify_question,
  {
    "stock_specific": "handle_stock_specific",
    "comparison": "handle_comparison",
    "technical_analysis": "handle_technical"
  }
)
workflow.add_edge("handle_stock_specific", END)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_comparison", END)

app = workflow.compile(checkpointer=memory)


def create_workflow():
    return app



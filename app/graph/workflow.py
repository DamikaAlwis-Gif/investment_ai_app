from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from .graph_state import GraphState
from .nodes import extract_context, handle_comparison, handle_stock_specific
from .nodes import handle_technical, summarize_conversation, formulate_query, generate, retreive
from .edges import classify_question

memory = MemorySaver()

workflow = StateGraph(GraphState)

workflow.add_node("formulate_query", formulate_query)
workflow.add_node("extract_context", extract_context)
workflow.add_node("handle_comparison", handle_comparison)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_stock_specific", handle_stock_specific)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("retreive", retreive)
workflow.add_node("generate", generate)


workflow.add_edge(
  START, "formulate_query"
)
workflow.add_edge(
  "formulate_query", "extract_context"
)

workflow.add_conditional_edges(
  "extract_context",
  classify_question,
  {
    "stock_specific": "handle_stock_specific",
    "comparison": "handle_comparison",
    "technical_analysis": "handle_technical",
    "news_based" : "retreive"
  }
)
workflow.add_edge("retreive", "generate")
workflow.add_edge("generate", "summarize_conversation")
workflow.add_edge("handle_stock_specific","summarize_conversation" )
workflow.add_edge("handle_technical","summarize_conversation" )
workflow.add_edge("handle_comparison","summarize_conversation" )

workflow.add_edge("summarize_conversation",END)

app = workflow.compile(checkpointer=memory)


def create_workflow():
    return app



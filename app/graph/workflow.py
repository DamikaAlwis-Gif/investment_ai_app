from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from .graph_state import GraphState
from .nodes import extract_context, get_question_type
from .nodes import stock_analysis, summarize_conversation, formulate_query, news_analysis, retreive
from .edges import classify_question
from .tools import retrive_technical_analysis_stock_data, retrieve_single_stock_data, retrieve_multi_stocks_data

memory = MemorySaver()

workflow = StateGraph(GraphState)

workflow.add_node("formulate_query", formulate_query)
workflow.add_node("extract_context", extract_context)
workflow.add_node("retrieve_multi_stocks_data", retrieve_multi_stocks_data)
workflow.add_node("retrive_technical_analysis_stock_data",
                  retrive_technical_analysis_stock_data)
workflow.add_node("retrieve_single_stock_data", retrieve_single_stock_data)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("retreive_news_data", retreive)
workflow.add_node("news_analysis", news_analysis)
workflow.add_node("stock_analysis", stock_analysis)
workflow.add_node("get_question_type", get_question_type)

workflow.add_edge(
  START, "formulate_query"
)
workflow.add_edge(
  "formulate_query", "extract_context"
)

workflow.add_edge(
  "extract_context",
  "get_question_type")
  

# workflow.add_conditional_edges(
#   "get_question_type",
#   classify_question,
#   {
#       "stock_specific": "retrieve_single_stock_data",
#       "comparison": "retrieve_multi_stocks_data",
#       "technical_analysis": "retrive_technical_analysis_stock_data",
#       "news_based": "retreive_news_data"
#   }
# )

workflow.add_conditional_edges(
    "get_question_type",
    lambda x :{
        "stock_specific": "retrieve_single_stock_data",
        "comparison": "retrieve_multi_stocks_data",
        "technical_analysis": "retrive_technical_analysis_stock_data",
        "news_based": "retreive_news_data"
    }[x["question_type"]]
)

workflow.add_edge("retreive_news_data", "retrieve_multi_stocks_data")
workflow.add_conditional_edges(
    "retrieve_multi_stocks_data",
    lambda x: {
        "comparison": "stock_analysis",
        "news_based": "news_analysis"
    }[x["question_type"]]
)

workflow.add_edge("retrive_technical_analysis_stock_data", "stock_analysis")
workflow.add_edge("retrieve_single_stock_data", "stock_analysis")
workflow.add_edge("news_analysis", "summarize_conversation")
workflow.add_edge("stock_analysis", "summarize_conversation")
workflow.add_edge("summarize_conversation",END)

app = workflow.compile(checkpointer=memory)


def create_workflow():
    return app



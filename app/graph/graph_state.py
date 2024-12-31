from typing_extensions import TypedDict
from typing import Annotated, Sequence, List, Dict, Any, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


# class GraphState(TypedDict):
#   input: str  # user input
#   chat_history: Annotated[Sequence[BaseMessage], add_messages]  # chat history
#   summary: str  # the summary of the chat history
#   answer: str  # answer genarated by the model
#   formatted_query: str  # standalone query
#   vector_store_documents: Sequence[Document]  # retrived vector store documents
#   web_search_results: Sequence[Document]  # retrived web search results

# Define question types
QuestionType = Literal[
    "stock_specific",
    "market_trend",
    "investment_strategy",
    "news_based",
    "comparison",
    "technical_analysis"
]



class GraphState(TypedDict):
    input : str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # question_type: QuestionType
    # stocks: Sequence[str]
    analysis_results: Dict
    context: Dict
    summary : str
    formatted_query : str
    vector_store_documents: Sequence[Document]

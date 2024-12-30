from .graph_state import GraphState
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from .chains import get_extract_context_chain, get_handle_stock_specific_chain, get_handle_comparison_chain, get_handle_technical_chain, get_formulated_query_chain
from .tools import analyze_single_stock, compare_stocks, technical_analysis
from .errors.finance_exceptions import MissingStockSymbolError, InsufficientStockSymbolsError
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

def extract_context(state: GraphState):
    """Extract relevant context like stock symbols from the question"""

    print("Extracting context...")
    context_chain = get_extract_context_chain()
    context = context_chain.invoke({
        "question": state["formatted_query"]
    })
    print("Context extracted:", context)
    return {
        "context": context,
    }


def handle_stock_specific(state: GraphState):
    """Handle questions about specific stocks"""

    symbol = state["context"]["symbols"][0] if state["context"]["symbols"] else None
    if not symbol:
        raise MissingStockSymbolError()

    analysis = analyze_single_stock(symbol)
    print(analysis)
    chain = get_handle_stock_specific_chain()
    response = chain.invoke({
        "symbol": symbol,
        "analysis": analysis
    })
    return {
        "messages": [AIMessage(response)]
    }


def handle_comparison(state: GraphState):
    """Handle stock comparison questions"""
    symbols = state["context"]["symbols"]

    if not symbols or len(symbols) < 2:
        raise InsufficientStockSymbolsError()

    comparison = compare_stocks(symbols)

    chain = get_handle_comparison_chain()

    response = chain.invoke({
        "comparison": comparison
    })
    return {
        "messages": [AIMessage(response)]
    }


def handle_technical(state: GraphState):
    """Handle technical analysis questions"""

    symbol = state["context"]["symbols"][0] if state["context"]["symbols"] else None
    if not symbol:
        raise MissingStockSymbolError()

    analysis = technical_analysis(symbol)
    chain = get_handle_technical_chain()
    response = chain.invoke({
        "symbol": symbol,
        "analysis": analysis
    })

    return {
        "messages": [AIMessage(response)]
    }


def summarize_conversation(state: GraphState):
  print("---Summarize Conversation---")
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

  summary = state.get("summary", "")
  if summary:
    summary_message = """
    This is summary of the conversation to date: {summary}
    Extend the summary by taking into account the new messages above.
    """

  else:
    summary_message = "Create a summary of the conversation above."
  messages = state["messages"] + [HumanMessage(summary_message)]
  chain = llm | StrOutputParser()
  summary = chain.invoke(messages)

  delete_messages = [RemoveMessage(id=m.id)
                     for m in state["messages"][:-4]]

  return {"summary": summary, "messages": delete_messages}


def formulate_query(state: GraphState):

  chain = get_formulated_query_chain()

  chat_history = state["messages"]
  input = state["input"]
  summary = state.get("summary", "")

  print("Chat History:", chat_history)
  print("summary:", summary)
  
  if not chat_history:
     formatted_query = input
  else:
    formatted_query = chain.invoke(
        {
            "chat_history": chat_history,
            "input": input,
            "summary": summary,
        }
    )

  print("Formatted Query:", formatted_query)
  print("Orginal query:", input)
  return {
     "formatted_query": formatted_query,
     "messages": [HumanMessage(state["input"])],
     "summary": summary, }

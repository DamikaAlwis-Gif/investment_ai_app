from .graph_state import GraphState
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from .chains import get_extract_context_chain, get_handle_stock_specific_chain, get_handle_comparison_chain, get_handle_technical_chain
from .tools import analyze_single_stock, compare_stocks, technical_analysis
from .errors.finance_exceptions import MissingStockSymbolError, InsufficientStockSymbolsError


def extract_context(state: GraphState):
    """Extract relevant context like stock symbols from the question"""

    print("Extracting context...")
    context_chain = get_extract_context_chain()
    context = context_chain.invoke({
        "question": state["input"]
    })
    print("Context extracted:", context)
    return {
        "context": context,
        "messages" : [HumanMessage(state["input"])]
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

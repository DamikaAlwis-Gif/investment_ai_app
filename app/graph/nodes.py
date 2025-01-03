from .graph_state import GraphState
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from .chains import get_extract_context_chain, get_classify_question_chain
from .chains import get_formulated_query_chain, get_rag_chain, get_handle_stock_analysis_chain
from .errors.finance_exceptions import MissingStockSymbolError, InsufficientStockSymbolsError
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from utils.retriver import get_retriever
from utils.vector_store import get_vector_store
from langchain_core.documents import Document

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


def stock_analysis(state: GraphState):
  
  chain = get_handle_stock_analysis_chain()

  print("Stock analysis...")

  response = chain.invoke({
        "formulated_question": state["formatted_query"],
        "original_question" : state["input"],
        "data": state["analysis_results"]
    })
  
  return {
        "messages": [AIMessage(response)]
    }



def summarize_conversation(state: GraphState):
  """Summarize a conversation and keep only limited messages in history"""
  
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
  # print(messages)
  chain = llm | StrOutputParser()
  summary = chain.invoke(messages)
  # print(summary)

  delete_messages = [RemoveMessage(id=m.id)
                     for m in state["messages"][:-4]]

  return {
    "summary": summary, 
    "messages": delete_messages,
    "analysis_results" : {},
    "vector_store_documents" : []
    }


def formulate_query(state: GraphState):
  """Genarate a standalone query based on original query, summary and chat history"""
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


def retreive(state: GraphState):
  ''' Retrive related documents using the query'''
  print("----Retrive----")
  question = state["formatted_query"]

  vectorstore = get_vector_store()

  documents = vectorstore.as_retriever(
     search_type="mmr",
     search_kwargs={'k': 4, 'fetch_k': 10}
  ).invoke(
     question
  )
  print(len(documents))
  vector_store_documents = []
  for doc in documents:
    page_content = doc.page_content
    metadata = doc.metadata
    new_metadata = {key: value for key,
                    value in metadata.items() if key != "embedding"}
    
    vector_store_documents.append(Document(page_content= page_content, metadata = new_metadata))



  print(vector_store_documents)
  return {"vector_store_documents": vector_store_documents}


def news_analysis(state: GraphState):
  # get vector store documents
  vector_store_documents = state.get("vector_store_documents", [])
  stock_analysis_results = state.get("stock_analysis_results", [])
  rag_chain = get_rag_chain()
  
  # fucntion to call rag chain invoke
  
  response = rag_chain.invoke(
        {
            "input": state["input"],
            "formatted_query": state["formatted_query"],
            "context": {
              "news_data": vector_store_documents,
              "stock_data": stock_analysis_results
              
            }
        }
  )
   
  
  return {"messages": [AIMessage(response)]}


def get_question_type(state: GraphState):
  print("Classifying Question...")
  classify_chain = get_classify_question_chain()

  response = classify_chain.invoke(
      {
          "question": state["formatted_query"]
      }
  )
  question_category = response.category
  
  return {
     "question_type" : question_category
  }

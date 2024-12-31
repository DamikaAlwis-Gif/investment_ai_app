from typing import Literal
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_groq import ChatGroq
# from langchain_google_genai import GoogleGenerativeAI
# from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI

def get_classify_question_chain():

  llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
  class QuestionCategory(BaseModel):

    category: Literal[
        "stock_specific",
        # "market_trend",
        # "investment_strategy",
        "news_based",
        "comparison",
        "technical_analysis"
    ] = Field(
      ...,
      description=(
       
          """
          Question category (stock_specific, news_based, comparison, technical_analysis)
          """
      )
    )

  system_message ='''
        Your are question classification expert. You shoulf classify 
        the following investment-related question into one of these categories:
        - stock_specific: Questions about a single stock
        - comparison: Questions comparing multiple stocks
        - technical_analysis: Questions about specific stock mentioning financial metrices
        - news_based: Questions about news
        Respond with only the category name.
  '''
  question_classify_prompt = ChatPromptTemplate.from_messages(
      [
        ("system" , system_message),
        ("human", "Question : {question}")
      ]
    )
  
  return question_classify_prompt | llm.with_structured_output(QuestionCategory)


def get_extract_context_chain():

  llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
  
  class Context(BaseModel):
    symbols : list[str] = Field(
        description="Stock symbols mentioned (in uppercase). Empty list if no symbols are found."
    )
    period : str = Field(
        description="Time period for analysis. Empty string if no period is mentioned.",
        examples=['1d', '5d', '1mo', '3mo', '6mo',
                  '1y', '2y', '5y', '10y', 'ytd', 'max']
    )
    metrics : list[str] = Field(
        description="Requested metrics. Empty string if no metrics are requested."
    )
    

  parser = JsonOutputParser(pydantic_object=Context)

  prompt = PromptTemplate(
    template= """
        You are a financial assistant. Analyze the following question and extract:
        1. Stock symbols mentioned (in uppercase). If company names are mentioned instead of stock symbols, resolve them to their stock symbols.
        2. Time period mentioned. Choose the most suitable one from the following options: ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'].
        3. Any specific metrics requested
        
        If you cannot find:
        - A stock symbol or company name, return an empty list for "symbols".
        - A time period, return an empty string for "period".
        - Specific metrics, return an empty list for "metrics".
        Question: {question}
        Format instructions: {format_instructions}
      """,
      input_variables=["question"],
      partial_variables={"format_instructions" :parser.get_format_instructions()}

      
  )
  return prompt | llm | parser



def get_handle_stock_analysis_chain():

  llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
  # llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest",
  #                          temperature=0.5)
  prompt = PromptTemplate(
      template="""
    You are a financial data analyst expert.
    Based on the user's question: {question}
    Provide a detailed and insightful answer, utilizing the following data:
    Data :  {data}
    Use a professional tone and use tables if applicable to present information clearly.
    """,
    input_variables=["question", "data"]
  )

  return prompt | llm | StrOutputParser()



def get_formulated_query_chain():

  contextualize_q_system_prompt = """
  You are an AI assistant that can formulate a standalone question 
  when given chat history, chat summary and the latest user question 
  which might reference context in the chat history and summary.
  Formulate a standalone question 
  which can be understood without the chat history. Do NOT answer the question, 
  just reformulate it if needed and otherwise return it as is.
  Output only the standalone question or the original if no reformulation is needed.
  """

  prompt_genrate_q = ChatPromptTemplate(
      [
          ("system", contextualize_q_system_prompt),
          ("human", "User question: {input} \n\n Chat summary: {summary}"),
          MessagesPlaceholder("chat_history"),
      ]
  )

  llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest",
                           temperature=0,)

  chain = prompt_genrate_q | llm | StrOutputParser()
  return chain


def get_rag_chain():

  system = """
    You are an financial domain expert for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    The Context may include stock information and related news information. 
    If you don't know the answer, just say that you don't know. 
    Please attach the URL of the relevant news articles or sources to your answer. 
    If the original user query is not clear, use the formulated query to answer the question. 
    Give answers based on the context only.
    
    When providing the answer:
    - First, answer the question.
    - Lastly, provide the sources in the following format:
    Sources:
    <The title> 
    <The URL or citation goes here>
  """

  rag_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system),
          ("human",
           "Context: {context} \n\nUser question: {input} \n\nFormulated question: {formatted_query}")
      ]
  )

  # llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
  llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                 temperature=0.5,)

  rag_chain = rag_prompt | llm | StrOutputParser()
  return rag_chain

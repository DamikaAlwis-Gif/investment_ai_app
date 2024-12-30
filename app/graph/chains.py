from typing import Literal
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_groq import ChatGroq
# from langchain_google_genai import GoogleGenerativeAI
# from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI

def get_classify_question_chain():

  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
  class QuestionCategory(BaseModel):

    category: Literal[
        "stock_specific",
        # "market_trend",
        # "investment_strategy",
        # "news_based",
        "comparison",
        "technical_analysis"
    ] = Field(
      ...,
      description=(
       
          """
          Question category (stock_specific, market_trend, investment_strategy, news_based, comparison, technical_analysis)
          """
      )
    )

  system_message ='''
        Classify the following investment-related question into one of these categories:
        - stock_specific: Questions about a single stock
        - comparison: Questions comparing multiple stocks
        - technical_analysis: Questions about technical indicators
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

  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
  
  class Context(BaseModel):
    symbols : list[str] = Field(
        description="Stock symbols mentioned (in uppercase). Empty list if no symbols are found.",
    )
    period : str = Field(
        description="Time period for analysis. Empty string if no period is mentioned."
    )
    metrics : list[str] = Field(
        description="Requested metrics. Empty string if no metrics are requested."
    )

  parser = JsonOutputParser(pydantic_object=Context)

  prompt = PromptTemplate(
    template= """
        You are a financial assistant. Analyze the following question and extract:
        1. Stock symbols mentioned (in uppercase).
        2. Time period mentioned
        3. Any specific metrics requested
        
        If you cannot find:
        - A stock symbol, return an empty list for "symbols".
        - A time period, return an empty string for "period".
        - Specific metrics, return an empty list for "metrics".
        Question: {question}
        Format instructions: {format_instructions}
      """,
      input_variables=["question"],
      partial_variables={"format_instructions" :parser.get_format_instructions()}

      
  )
  return prompt | llm | parser


def get_handle_stock_specific_chain():

  # llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
  llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest",
                           temperature=0,)

  prompt = PromptTemplate(
      template="""
      Provide a detailed analysis for {symbol} based on this data:
      {analysis}

      Include:
      1. Current market position
      2. Key metrics interpretation
      3. Notable strengths/weaknesses
      4. Investment considerations
      U may use graphs, tables to preset the information.
      """,
      input_variables=["symbol", "analysis"]
    )
  return prompt | llm | StrOutputParser()

def get_handle_comparison_chain():
 
  # llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
  llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest",
                           temperature=0,)
  prompt = PromptTemplate(
      template="""
      Compare these stocks based on the following data:
        {comparison}
        
        Include:
        1. Head-to-head metric comparison
        2. Relative strengths/weaknesses
        3. Investment recommendation
      U may use graphs, tables to preset the information.  
      """,
      input_variables=["comparison"]
  )
  return prompt | llm | StrOutputParser()

def get_handle_technical_chain(): 
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
  prompt = PromptTemplate(
      template="""
      Provide technical analysis for {symbol} based on these indicators:
        {analysis}
        
        Include:
        1. Trend interpretation
        2. Support/resistance levels
        3. RSI analysis
        4. Volume analysis
        5. Trading signals
      Provide the information in a visual appleaing way.
      """,
      input_variables=["analysis", "symbol"],
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

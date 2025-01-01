from .graph_state import GraphState
from .chains import get_classify_question_chain


def classify_question(state : GraphState):
  print("Classifying Question...")
  classify_chain = get_classify_question_chain()
  
  response = classify_chain.invoke(
    {
        "question": state["formatted_query"]
    }
  )
  question_category = response.category
  print(f"Question category : {question_category}")
  return question_category




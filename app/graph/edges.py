from .graph_state import GraphState
from .chains import get_classify_question_chain


def classify_question(state : GraphState):
  return state['question_type']




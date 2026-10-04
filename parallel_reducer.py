
#                  AnalyzerState
#                        │
#               ┌────────┴────────┐
#               │                 │
#           raw_text         safety_score
#               │                 │
#             string          dictionary
#                                 │
#                        ┌────────┴────────┐
#                        │                 │
#                      key               value
#                     string             integer
#                        │
#                "toxicity_score"
#                "cultural_score"
#                "copyright_score"

# "This field contains a dictionary, and whenever this field gets multiple updates, use merge_score_dicts to combine them."

# Imagine you have two LangGraph nodes.
# Node 1 gives:
# {    "safety_score": {        "toxicity_score": 10    }}


# Node 2 gives:
# {    "safety_score": {        "cultural_score": 20    }}


# LangGraph sees:
# safety_score
#      ↓
# Multiple updates
#      ↓
# merge_score_dicts()
#      ↓
# Combine them

# Result:
# {    "toxicity_score": 10,    "cultural_score": 20}



from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# INITIALIZE LLM
# ============================================================

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1000,
)


# ============================================================
# FUNCTION TO MERGE DICTIONARIES
# ============================================================

def merge_score_dicts(
    existing: dict | None,
    newupdate: dict
) -> dict:

    if existing is None:
        return newupdate

    return {**existing, **newupdate}


# ============================================================
# CREATE STATE
# ============================================================

class AnalyzerState(TypedDict):
    raw_text: str
     #reducer function is used to merge the values of the same key 
    safety_score: Annotated[
        dict[str, int],
        merge_score_dicts
    ]


# ============================================================
# NODE 1: TOXICITY
# ============================================================

def toxicity_node(state: AnalyzerState) -> dict:

    print("\n[Branch 1] Analyzing toxicity and hate speech...")

    prompt = f"""
Analyze the following text for profanity, aggression,
hate speech, or toxicity.

Provide a score from 0 to 100.

0 = perfectly clean
100 = extremely toxic

Return ONLY a single integer number. Nothing else.

Text:
{state['raw_text']}
"""

    response = llm.invoke(prompt)

    score = int(response.content.strip())

    return {
        "safety_score": {
            "toxicity_score": score
        }
    }


# ============================================================
# NODE 2: CULTURAL SENSITIVITY
# ============================================================

def cultural_node(state: AnalyzerState) -> dict:

    print("\n[Branch 2] Analyzing cultural sensitivity...")

    prompt = f"""
Analyze the following text for cultural sensitivity.

Provide a score from 0 to 100.

0 = perfectly culturally appropriate
100 = extremely culturally inappropriate

Return ONLY a single integer number. Nothing else.

Text:
{state['raw_text']}
"""

    response = llm.invoke(prompt)

    score = int(response.content.strip())

    return {
        "safety_score": {
            "cultural_score": score
        }
    }


# ============================================================
# NODE 3: COPYRIGHT
# ============================================================

def copy_right_node(state: AnalyzerState) -> dict:

    print("\n[Branch 3] Checking for copyright infringement...")

    prompt = f"""
Analyze the following text for potential copyright
infringement or plagiarism.

Provide a score from 0 to 100.

0 = no copyright risk
100 = high copyright risk

Return ONLY a single integer number. Nothing else.

Text:
{state['raw_text']}
"""

    response = llm.invoke(prompt)

    score = int(response.content.strip())

    return {
        "safety_score": {
            "copyright_score": score
        }
    }


# ============================================================
# CREATE GRAPH
# ============================================================

builder = StateGraph(AnalyzerState)


# ============================================================
# ADD NODES
# ============================================================

builder.add_node("toxicity", toxicity_node)
builder.add_node("cultural_check", cultural_node)
builder.add_node("copyright", copy_right_node)


# ============================================================
# ADD EDGES
# ============================================================

# All three nodes start at the same time
builder.add_edge(START, "toxicity")
builder.add_edge(START, "cultural_check")
builder.add_edge(START, "copyright")


# All three nodes finish the graph
builder.add_edge("toxicity", END)
builder.add_edge("cultural_check", END)
builder.add_edge("copyright", END)


# ============================================================
# COMPILE GRAPH
# ============================================================

app = builder.compile()


# ============================================================
# INVOKE GRAPH
# ============================================================

input_data = {
    "raw_text": "I am going to kill you, you son of a bitch"
}

result = app.invoke(input_data)


# ============================================================
# PRINT RESULT
# ============================================================

print("\n=== FINAL SAFETY SCORES ===\n")

print(result["safety_score"])
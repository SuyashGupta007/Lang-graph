# INITIAL STATE
# │
# ├── program = "BCA"
# │
# └── messages
#       └── HumanMessage("What subjects do I have?")
#              │
#              ▼
#         CLASSIFIER
#              │
#        ┌─────┼─────┐
#        ▼     ▼     ▼
#    academic fees general
#        │     │     │
#        ▼     ▼     ▼
#    Academic Fee   General
#       RAG    RAG
#        │     │     │
#        └─────┼─────┘
#              ▼
#           RESPONSE
#              │
#              ▼
#             END



import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing in .env file")


# ============================================================
# 2. INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.4,
    max_tokens=1000
)


# ============================================================
# 3. INITIALIZE EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. FUNCTION TO CREATE A RETRIEVER FROM PDF
# ============================================================

def build_retriever(pdf_path: str):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"File not found: {pdf_path}"
        )

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # Split PDF into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        chunks,
        embedding_model
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever


# ============================================================
# 5. CREATE RETRIEVERS
# ============================================================

academic_retriever = build_retriever(
    "academics_handbook.pdf"
)

fee_retriever = build_retriever(
    "fee_structure.pdf"
)


# ============================================================
# 6. DEFINE LANGGRAPH STATE
# ============================================================

class State(TypedDict):

    # Student's program
    # BCA / BBA / B.Com
    program: str

    # Conversation messages
    messages: Annotated[list, add_messages]

    # Query category
    # academic / fees / general
    query_type: str

    # Retrieved PDF content
    retrieved_context: str


# ============================================================
# 7. CLASSIFIER NODE
# ============================================================

def classifier_node(state: State):

    program = state["program"]

    question = state["messages"][-1].content

    prompt = f"""
You are a university chatbot query classifier.

Student Program:
{program}

Classify the student's question into exactly ONE category:

academic
fees
general

Classification rules:

academic:
Questions about subjects, courses, syllabus, exams,
grades, attendance, curriculum, academic rules,
semester information, or academic requirements.

fees:
Questions about tuition fees, scholarships,
payments, fee deadlines, penalties, refunds,
or payment-related information.

general:
Anything that does not belong to academic or fees.

Student Question:
{question}

Return ONLY one of these three words:

academic
fees
general
"""

    response = llm.invoke(prompt)

    query_type = response.content.strip().lower()

    # Safety check
    if query_type not in [
        "academic",
        "fees",
        
    ]:
        query_type = "general"

    return {
        "query_type": query_type
    }


# ============================================================
# 8. ACADEMIC RAG NODE
# ============================================================

def academic_rag_node(state: State):

    program = state["program"]

    question = state["messages"][-1].content

    # Include program while searching
    search_query = f"""
Program: {program}

Question:
{question}
"""

    documents = academic_retriever.invoke(
        search_query
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# 9. FEE RAG NODE
# ============================================================

def fee_rag_node(state: State):

    program = state["program"]

    question = state["messages"][-1].content

    # Include program while searching
    search_query = f"""
Program: {program}

Question:
{question}
"""

    documents = fee_retriever.invoke(
        search_query
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# 10. GENERAL NODE
# ============================================================

def general_node(state: State):

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# ============================================================
# 11. RESPONSE NODE
# ============================================================

def response_node(state: State):

    program = state["program"]

    question = state["messages"][-1].content

    context = state["retrieved_context"]

    # --------------------------------------------------------
    # GENERAL QUESTION
    # --------------------------------------------------------

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = f"""
You are a helpful university chatbot.

Student Program:
{program}

Answer the student's question clearly and
concisely using general knowledge.

Student Question:
{question}
"""

    # --------------------------------------------------------
    # RAG QUESTION
    # --------------------------------------------------------

    else:

        prompt = f"""
You are a university chatbot.

Student Program:
{program}

Answer the student's question using ONLY
the context provided below.

Do not use outside information.

If the answer is not present in the context,
say:

"I cannot answer this based on the provided documents."

Student Question:
{question}

Context:
{context}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }


# ============================================================
# 12. ROUTER
# ============================================================

def route_query(state: State):

    query_type = state["query_type"]

    if query_type == "academic":

        return "academic_rag"

    elif query_type == "fees":

        return "fee_rag"

    else:

        return "general"


# ============================================================
# 13. CREATE STATE GRAPH
# ============================================================

graph_builder = StateGraph(State)


# ============================================================
# 14. ADD NODES
# ============================================================

graph_builder.add_node(
    "classifier",
    classifier_node
)

graph_builder.add_node(
    "academic_rag",
    academic_rag_node
)

graph_builder.add_node(
    "fee_rag",
    fee_rag_node
)

graph_builder.add_node(
    "general",
    general_node
)

graph_builder.add_node(
    "response",
    response_node
)


# ============================================================
# 15. ADD EDGES
# ============================================================

# START → CLASSIFIER

graph_builder.add_edge(
    START,
    "classifier"
)


# CLASSIFIER → ROUTER

graph_builder.add_conditional_edges(
    "classifier",
    route_query,
    {
        "academic_rag": "academic_rag",
        "fee_rag": "fee_rag",
        "general": "general"
    }
)


# RAG → RESPONSE

graph_builder.add_edge(
    "academic_rag",
    "response"
)

graph_builder.add_edge(
    "fee_rag",
    "response"
)

graph_builder.add_edge(
    "general",
    "response"
)


# RESPONSE → END

graph_builder.add_edge(
    "response",
    END
)


# ============================================================
# 16. COMPILE GRAPH
# ============================================================

app = graph_builder.compile()


# ============================================================
# 17. INITIAL STATE
# ============================================================

initial_state = {

    # Choose BCA, BBA or B.Com
    "program": "BCA",

    # Initial human message
    "messages": [
        HumanMessage(
            content="What subjects do I have in semester 3?"
        )
    ],

    # Initially empty
    "query_type": "",

    # Initially empty
    "retrieved_context": ""
}


# ============================================================
# 18. RUN GRAPH
# ============================================================

result = app.invoke(
    initial_state
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print("\n======================================")
print("        UNIVERSITY CHATBOT")
print("======================================")

print(
    "\nProgram:",
    result["program"]
)

print(
    "\nQuestion:",
    result["messages"][0].content
)

print(
    "\nQuery Type:",
    result["query_type"]
)

print(
    "\nAnswer:",
    result["messages"][-1].content
)

print("\n======================================")
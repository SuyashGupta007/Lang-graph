import os
from typing import TypedDict, Annotated

import streamlit as st
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
# PAGE CONFIG (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="University Chatbot",
    page_icon="🎓",
    layout="centered",
)

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file and restart.")
    st.stop()

ACADEMIC_PDF = "academics_handbook.pdf"
FEE_PDF = "fee_structure.pdf"

for pdf in (ACADEMIC_PDF, FEE_PDF):
    if not os.path.exists(pdf):
        st.error(f"File not found: {pdf}. Place it next to app.py.")
        st.stop()


# ============================================================
# BACKEND (cached so it is built only once, not on every rerun)
# ============================================================

class State(TypedDict):
    program: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


@st.cache_resource(show_spinner="Loading models and documents...")
def build_graph():
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.4,
        max_tokens=1000,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    def build_retriever(pdf_path: str):
        documents = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    academic_retriever = build_retriever(ACADEMIC_PDF)
    fee_retriever = build_retriever(FEE_PDF)

    # ---------------- Classifier ----------------
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

        if query_type not in ["academic", "fees", "general"]:
            query_type = "general"

        return {"query_type": query_type}

    # ---------------- RAG nodes ----------------
    def academic_rag_node(state: State):
        search_query = f"Program: {state['program']}\n\nQuestion:\n{state['messages'][-1].content}"
        documents = academic_retriever.invoke(search_query)
        context = "\n\n".join(d.page_content for d in documents)
        return {"retrieved_context": context}

    def fee_rag_node(state: State):
        search_query = f"Program: {state['program']}\n\nQuestion:\n{state['messages'][-1].content}"
        documents = fee_retriever.invoke(search_query)
        context = "\n\n".join(d.page_content for d in documents)
        return {"retrieved_context": context}

    def general_node(state: State):
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    # ---------------- Response ----------------
    def response_node(state: State):
        program = state["program"]
        question = state["messages"][-1].content
        context = state["retrieved_context"]

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
        return {"messages": [("ai", response.content.strip())]}

    # ---------------- Router ----------------
    def route_query(state: State):
        if state["query_type"] == "academic":
            return "academic_rag"
        elif state["query_type"] == "fees":
            return "fee_rag"
        return "general"

    # ---------------- Graph ----------------
    builder = StateGraph(State)
    builder.add_node("classifier", classifier_node)
    builder.add_node("academic_rag", academic_rag_node)
    builder.add_node("fee_rag", fee_rag_node)
    builder.add_node("general", general_node)
    builder.add_node("response", response_node)

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier",
        route_query,
        {
            "academic_rag": "academic_rag",
            "fee_rag": "fee_rag",
            "general": "general",
        },
    )
    builder.add_edge("academic_rag", "response")
    builder.add_edge("fee_rag", "response")
    builder.add_edge("general", "response")
    builder.add_edge("response", END)

    return builder.compile()


graph_app = build_graph()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    program = st.selectbox("Your program", ["BCA", "BBA", "B.Com"])

    show_details = st.toggle("Show query type & retrieved context", value=True)

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

    st.caption(
        "Routes each question to the Academic handbook, "
        "the Fee structure, or general knowledge."
    )


# ============================================================
# CHAT STATE
# ============================================================

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: role, content, query_type, context

BADGES = {
    "academic": "📚 Academic",
    "fees": "💰 Fees",
    "general": "💬 General",
}


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and show_details:
            st.caption(f"Category: {BADGES.get(msg.get('query_type'), '-')}")

            context = msg.get("context", "")
            if context and context != "NO_RETRIEVAL_NEEDED":
                with st.expander("📄 Retrieved context"):
                    st.text(context)


# ============================================================
# MAIN UI
# ============================================================

st.title("🎓 University Chatbot")
st.caption(f"Asking as a **{program}** student")

if not st.session_state.chat:
    st.info(
        "Try: *What subjects do I have in semester 3?* "
        "or *What is the tuition fee?*"
    )

for msg in st.session_state.chat:
    render_message(msg)

question = st.chat_input("Ask about subjects, fees, or anything else...")

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.chat.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph_app.invoke(
                    {
                        "program": program,
                        "messages": [HumanMessage(content=question)],
                        "query_type": "",
                        "retrieved_context": "",
                    }
                )
                answer = result["messages"][-1].content
                query_type = result["query_type"]
                context = result["retrieved_context"]
            except Exception as e:
                answer = f"Something went wrong: `{e}`"
                query_type = "general"
                context = ""

    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "query_type": query_type,
        "context": context,
    }
    st.session_state.chat.append(assistant_msg)
    st.rerun()
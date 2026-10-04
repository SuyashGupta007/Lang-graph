# writer → tools → writer → extract_draft → human_review (PAUSE)
#                                                │
#                                   ┌────────────┴────────────┐
#                                   ↓                         ↓
#                                approve                   revise
#                                   ↓                         ↓
#                                  END                      WRITER
#                                   ↑                         ↓
#                             resume graph               revise…

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. TOOL
# ============================================================

search_tool = TavilySearch(max_results=3)

tools = [search_tool]


# ============================================================
# 3. LLM
# ============================================================

writer_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)

writer_llm_with_tools = writer_llm.bind_tools(tools)


# ============================================================
# 4. STATE
# ============================================================

class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    human_feedback: str
    attempt: int


# ============================================================
# 5. WRITER PROMPT
# ============================================================

WRITER_SYSTEM_PROMPT = """
You are an advanced LinkedIn post writer.

Your job is to create high-quality LinkedIn posts from
the user's topic.

Rules:

- Start with a strong hook.
- Use short paragraphs.
- Keep the post professional and conversational.
- Make the post authentic and human.
- Do not invent facts or achievements.
- Use web search when current information is required.
- Use 3-5 relevant hashtags.
- Avoid excessive emojis.
- Avoid clickbait.
- Avoid generic motivational language.

If you receive human feedback, revise the post according
to that feedback.

Return ONLY the LinkedIn post.
"""


# ============================================================
# 6. WRITER NODE
# ============================================================

def writer_node(state: State):

    attempt = state.get("attempt", 0) + 1

    topic = state["topic"]

    human_feedback = state.get("human_feedback", "")

    # First attempt
    if attempt == 1:

        user_message = f"""
Write a LinkedIn post about:

{topic}

Use web search if current information is required.
"""

    # Revision after human feedback
    else:

        user_message = f"""
Rewrite the LinkedIn post.

Topic:
{topic}

Human feedback:
{human_feedback}

Apply the feedback carefully.

Do not change the main topic.

Return only the revised LinkedIn post.
"""

    response = writer_llm_with_tools.invoke([
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", user_message)
    ])

    return {
        "messages": [response],
        "attempt": attempt
    }


# ============================================================
# 7. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 8. WRITER ROUTER
# ============================================================

def writer_router(state: State):

    last_message = state["messages"][-1]

    # Writer wants to use Tavily
    if hasattr(last_message, "tool_calls"):

        if last_message.tool_calls:
            return "tools"

    # No tool call
    return "extract_draft"


# ============================================================
# 9. EXTRACT DRAFT
# ============================================================

def extract_draft_node(state: State):

    last_message = state["messages"][-1]

    draft = last_message.content

    print("\n")
    print("=" * 50)
    print("GENERATED LINKEDIN POST")
    print("=" * 50)
    print(draft)
    print("=" * 50)

    return {
        "draft": draft
    }


# ============================================================
# 10. HUMAN REVIEW NODE
# ============================================================

def human_review_node(state: State):

    draft = state["draft"]

    # --------------------------------------------------------
    # INTERRUPT
    # --------------------------------------------------------

    human_response = interrupt({
        "message": "Please review the LinkedIn post.",
        "draft": draft
    })

    # Whatever we pass in Command(resume=...)
    # comes back here.
    return {
        "human_feedback": human_response
    }


# ============================================================
# 11. HUMAN ROUTER
# ============================================================

def human_router(state: State):

    feedback = state["human_feedback"]

    # --------------------------------------------------------
    # APPROVE
    # --------------------------------------------------------

    if feedback.lower() == "approve":

        return "end"

    # --------------------------------------------------------
    # REVISE
    # --------------------------------------------------------

    return "writer"


# ============================================================
# 12. BUILD GRAPH
# ============================================================

graph = StateGraph(State)


# Add nodes

graph.add_node(
    "writer",
    writer_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "extract_draft",
    extract_draft_node
)

graph.add_node(
    "human_review",
    human_review_node
)


# ============================================================
# 13. EDGES
# ============================================================

# START → WRITER

graph.add_edge(
    START,
    "writer"
)


# WRITER → TOOLS / EXTRACT

graph.add_conditional_edges(
    "writer",
    writer_router,
    {
        "tools": "tools",
        "extract_draft": "extract_draft"
    }
)


# TOOLS → WRITER

graph.add_edge(
    "tools",
    "writer"
)


# EXTRACT → HUMAN

graph.add_edge(
    "extract_draft",
    "human_review"
)


# HUMAN → WRITER / END

graph.add_conditional_edges(
    "human_review",
    human_router,
    {
        "writer": "writer",
        "end": END
    }
)


# ============================================================
# 14. CHECKPOINTER
# ============================================================

memory = MemorySaver()


# Compile graph

app = graph.compile(
    checkpointer=memory
)


# ============================================================
# 15. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter LinkedIn post topic: "
    )

    # Initial state

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "human_feedback": "",
        "attempt": 0
    }

    # Thread ID
    # Important for resuming the paused graph

    config = {
        "configurable": {
            "thread_id": "linkedin-post-1"
        }
    }


    # ========================================================
    # FIRST RUN
    # ========================================================

    result = app.invoke(
        initial_state,
        config=config
    )


    # ========================================================
    # GRAPH IS NOW PAUSED AT interrupt()
    # ========================================================

    print("\n")
    print("=" * 50)
    print("HUMAN REVIEW REQUIRED")
    print("=" * 50)

    print(result)


    # ========================================================
    # ASK HUMAN
    # ========================================================

    print("\nWhat do you want to do?")

    print("1. approve")
    print("2. revise")

    decision = input(
        "\nEnter your decision: "
    ).strip().lower()


    # ========================================================
    # RESUME GRAPH
    # ========================================================

    result = app.invoke(
        Command(
            resume=decision
        ),
        config=config
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 50)
    print("FINAL RESULT")
    print("=" * 50)

    print(result.get("draft"))
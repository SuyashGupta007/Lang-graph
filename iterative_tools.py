import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. SEARCH TOOL
# ============================================================

search_tool = TavilySearch(max_results=3)

tools = [search_tool]


# ============================================================
# 3. WRITER LLM
# ============================================================

writer_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)

# Give the writer access to Tavily
writer_llm_with_tools = writer_llm.bind_tools(tools)


# ============================================================
# 4. REVIEWER LLM
# ============================================================

reviewer_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.2
)


# ============================================================
# 5. STATE
# ============================================================

class State(TypedDict):
    topic: str

    messages: Annotated[list, add_messages]

    draft: str

    review_feedback: str

    is_approved: bool

    attempt: int


# ============================================================
# 6. WRITER SYSTEM PROMPT
# ============================================================

WRITER_SYSTEM_PROMPT = """
You are an advanced LinkedIn post writer, personal-branding expert,
and research assistant.

Your job is to transform the user's ideas, experiences, achievements,
technical learnings, opinions, or professional updates into engaging,
high-quality LinkedIn posts.

You have access to a web search tool.

Use web search when:
- Current information is required.
- The user asks about recent technology or AI developments.
- The user asks about latest news or trends.
- A factual claim needs verification.
- The user explicitly asks for research.

Do NOT use web search when:
- The post is purely about personal experience.
- The user only wants rewriting.
- Current information is not necessary.

Writing rules:

1. Start with a strong hook.

2. Keep the post focused on one main idea.

3. Use short paragraphs.

4. Make the post easy to read on mobile.

5. Sound professional, authentic, conversational,
   thoughtful and human.

6. Do not invent:
   - achievements
   - statistics
   - companies
   - projects
   - experiences
   - facts

7. Preserve the user's meaning and voice.

8. For technical posts:
   - Keep technical information accurate.
   - Explain difficult concepts simply.
   - Use practical examples when useful.
   - Avoid unnecessary jargon.

9. Use storytelling when appropriate:

   Hook
   ↓
   Context
   ↓
   Problem
   ↓
   Experience
   ↓
   Learning
   ↓
   Takeaway
   ↓
   Closing thought

10. Use emojis sparingly.

11. Use 3-5 relevant hashtags.

12. Avoid:
   - fake motivational language
   - clickbait
   - corporate jargon
   - fake statistics
   - excessive emojis
   - generic quotes
   - artificial engagement bait

13. End with a meaningful question only when appropriate.

14. If revising a post, carefully apply all reviewer feedback.

FINAL OUTPUT:

Return ONLY the LinkedIn post.

Do not explain your writing process.

Do not say:
"Here is your LinkedIn post."

Do not provide analysis.
"""


# ============================================================
# 7. WRITER NODE
# ============================================================

def writer_node(state: State) -> dict:
    """
    Creates the first LinkedIn post
    or rewrites the post based on reviewer feedback.
    """

    attempt = state.get("attempt", 0) + 1

    topic = state["topic"]

    previous_feedback = state.get("review_feedback", "")

    # --------------------------------------------------------
    # First attempt
    # --------------------------------------------------------

    if attempt == 1:

        user_message = f"""
Write a LinkedIn post on this topic:

{topic}

If current information is required,
use the web search tool first.
"""

    # --------------------------------------------------------
    # Revision
    # --------------------------------------------------------

    else:

        user_message = f"""
Rewrite the LinkedIn post based on the reviewer feedback.

Original topic:
{topic}

Reviewer feedback:
{previous_feedback}

Fix every issue mentioned by the reviewer.

Do not change the main topic.

Do not introduce fake information.

Return only the revised LinkedIn post.
"""

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", user_message)
    ]

    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "attempt": attempt
    }


# ============================================================
# 8. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 9. EXTRACT DRAFT
# ============================================================

def extract_draft_node(state: State) -> dict:
    """
    Gets the final writer response and saves it as draft.
    """

    last_message = state["messages"][-1]

    draft = last_message.content

    print("\n==============================")
    print("DRAFT")
    print("==============================")
    print(draft)

    return {
        "draft": draft
    }


# ============================================================
# 10. REVIEWER NODE
# ============================================================

REVIEWER_PROMPT = """
You are an expert LinkedIn editor.

Review the following LinkedIn post.

Check:

1. Hook
2. Clarity
3. Storytelling
4. Grammar
5. Professional tone
6. Authenticity
7. Technical accuracy
8. Readability
9. Engagement
10. Hashtags
11. Whether the post sounds AI-generated
12. Whether the post stays focused on one main idea

IMPORTANT:

If the post is good enough, approve it.

If improvements are required, provide clear and specific feedback.

At the end of your response, write exactly one of:

APPROVED

or

REVISE
"""


def reviewer_node(state: State) -> dict:

    draft = state["draft"]

    prompt = f"""
{REVIEWER_PROMPT}

LinkedIn Post:

{draft}
"""

    response = reviewer_llm.invoke(prompt)

    feedback = response.content

    print("\n==============================")
    print("REVIEW")
    print("==============================")
    print(feedback)

    # --------------------------------------------------------
    # Check approval
    # --------------------------------------------------------

    if "APPROVED" in feedback.upper():

        is_approved = True

    else:

        is_approved = False

    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }


# ============================================================
# 11. ROUTING AFTER WRITER
# ============================================================

def writer_router(state: State):

    last_message = state["messages"][-1]

    # If LLM wants to use a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:

        return "tools"

    # Otherwise go directly to draft extraction
    return "extract_draft"


# ============================================================
# 12. ROUTING AFTER REVIEWER
# ============================================================

def reviewer_router(state: State):

    # Post approved
    if state["is_approved"]:

        return "end"

    # Prevent infinite loop
    if state["attempt"] >= 3:

        print("\nMaximum attempts reached.")

        return "end"

    # Need revision
    return "writer"


# ============================================================
# 13. BUILD GRAPH
# ============================================================

graph = StateGraph(State)


# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------

graph.add_node("writer", writer_node)

graph.add_node("tools", tool_node)

graph.add_node(
    "extract_draft",
    extract_draft_node
)

graph.add_node(
    "reviewer",
    reviewer_node
)


# ------------------------------------------------------------
# START → WRITER
# ------------------------------------------------------------

graph.add_edge(
    START,
    "writer"
)


# ------------------------------------------------------------
# WRITER → TOOLS OR EXTRACT DRAFT
# ------------------------------------------------------------

graph.add_conditional_edges(
    "writer",
    writer_router,
    {
        "tools": "tools",
        "extract_draft": "extract_draft"
    }
)


# ------------------------------------------------------------
# TOOLS → WRITER
# ------------------------------------------------------------

graph.add_edge(
    "tools",
    "writer"
)


# ------------------------------------------------------------
# EXTRACT DRAFT → REVIEWER
# ------------------------------------------------------------

graph.add_edge(
    "extract_draft",
    "reviewer"
)


# ------------------------------------------------------------
# REVIEWER → WRITER OR END
# ------------------------------------------------------------

graph.add_conditional_edges(
    "reviewer",
    reviewer_router,
    {
        "writer": "writer",
        "end": END
    }
)


# ============================================================
# 14. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# 15. RUN GRAPH
# ============================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter the topic for your LinkedIn post: "
    )

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0
    }

    result = app.invoke(initial_state)

    print("\n\n================================")
    print("FINAL LINKEDIN POST")
    print("================================\n")

    print(result["draft"])

    print("\n================================")
    print("TOTAL ATTEMPTS:", result["attempt"])
    print("================================")
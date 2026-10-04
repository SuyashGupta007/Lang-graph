import os 
from typing import TypedDict

# Lets create the state first 

class pipelinestate(TypedDict):
    raw_input:str
    edited_text:str
    script_text:str
    final_output:str


# Now we need to create the nodes

from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the ChatGroq model
model = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.7, #randomness - higher values means more random
    max_tokens=1000, #limit on the number of tokens in the response
    # top_p=0.9, #nucleus sampling - limits the number of tokens considered for the response
    # echo=False #if true, the response will include the prompt
)

def editor_node(state:pipelinestate) -> dict:
   #docstring: tell the node what it does 
    """ Stage 1: Cleans up grammatical errors and typos and refines the tone. """
    prompt=(
        f"""
    You are a professional copyeditor.
    Your job is to clean up and refine the text provided by the user.
    Please fix any grammatical errors, spelling mistakes, or typos.
    Also, improve the clarity and flow of the text while maintaining the original meaning.
    Return only the edited text.\n\n
    
    Original Text:\n
    {state['raw_input']}\n\n
    
   
    """
    )
    response = model.invoke(prompt)
    return {"edited_text": response.content.strip()}


def scriptwriter_node(state:pipelinestate) -> dict:
   """ Stage 2: Expands the text into a detailed script for a video or presentation. """
   prompt=(
    f"""
    You are a scriptwriter.
    Expand the given text into a detailed script for a video or presentation.
    Add engaging elements like an introduction, body, and conclusion.
    Maintain the original meaning and tone.
    
    Edited Text:
    {state['edited_text']}
    
    Full Script:
    """
   )
   response = model.invoke(prompt)
   return {"script_text": response.content.strip()}


def translator_node(state:pipelinestate) -> dict:
   """ Stage 3: Translate the script into natural flowing Hinglish. """
   prompt=(
    f"""
    You are a professional translator.
    Translate the following English script into natural Hinglish (Hindi + English).
    Maintain the original meaning, tone, and structure.
    
   Script in English:
    {state['script_text']}
    
    Final Hinglish script:
    """
   )
   response = model.invoke(prompt)
   return {"final_output": response.content.strip()}

#now your state and nodes are ready and now it is time to create the graph
#and for creating the graph you have to connect tese nodes and for that you have 
#to use edges 
#edges are very imp to create the workflows

from langgraph.graph import StateGraph , START ,END

#create the graph
graph = StateGraph(pipelinestate)
#add nodes in graph 
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

#now connect the nodes to create the workflow
# Add edges(sequential - one after another )
graph.add_edge(START,"editor")
graph.add_edge("editor","scriptwriter")
graph.add_edge("scriptwriter","translator")
graph.add_edge("translator",END)

#complie the graph 

app = graph.compile()
# now this app can work as runnable
result = app.invoke({"raw_input": "Ai agents are the future of the tech .They can think , plan and act on their own.LangGraph is a tool that helps to create these agents with proper control and memeory. "}) 

#output
print("your result are:-\n")
print(result['final_output'])


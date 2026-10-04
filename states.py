#now we are going to create a graph

# and the first thing you create is a state 

import os 

#1 ) typed DICT(Most common approach)



from typing import TypedDict

class State(TypedDict):
    topic :str
    summary: str
    score:int


#2) Pydantic approach 


from pydantic import BaseModel,field_validator
#it is good at data validation and type checking at runtime
class State(BaseModel):
    topic:str
    summary:str = ""
    score:int
    
    @field_validator
    def score_positive(cls,v):
        if v<0:
            raise ValueError("score must be positive")
        return v


#3) dataclass approach 

from dataclasses import dataclass

@dataclass
class State:
    topic:str
    summary:str
    score:str


#4) MessageState approach 

from langgraph.graph import MessagesState

class State(MessagesState):
    #messages field is already included with add_messages reducer 
    # just add your extra fields
    topic:str
    summary:str
    score:int
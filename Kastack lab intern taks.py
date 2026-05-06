import os
import json
import pandas as pd
import numpy as np
from typing import TypedDict, List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=LLM_MODEL, 
    task="text-generation", 
    max_new_tokens=512, 
    temperature=0.3
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./rag_db")

class UserPersona(BaseModel):
    habits: list[str] = Field(description="User habits e.g., late sleeper, food preferences")
    personal_facts: list[str] = Field(description="Facts about the user, relationships, events")
    personality_traits: list[str] = Field(description="Adjectives describing personality")
    communication_style: list[str] = Field(description="Tone, length of messages, emoji usage")

persona_parser = JsonOutputParser(pydantic_object=UserPersona)

summary_prompt = PromptTemplate.from_template("Summarize this conversation segment chronologically:\n\n{messages}\n\nSummary:")
persona_prompt = PromptTemplate(
    input_variables=["text"],
    partial_variables={"format_instructions": persona_parser.get_format_instructions()},
    template="Extract the user persona from the following text.\n{format_instructions}\n\nText: {text}\n\nJSON Output:"
)
qa_prompt = PromptTemplate.from_template("""You are an AI assistant.
User Persona:
{persona}

Context:
{context}

Question: {question}
Answer using only context and persona:""")

class IngestionState(TypedDict):
    current_message: str
    topic_buffer: List[str]
    topic_embedding: Any
    hundred_msg_buffer: List[str]
    persona_data: List[Dict]
    topic_count: int
    checkpoint_count: int

def ingest_message(state: IngestionState):
    msg = state["current_message"]
    state["topic_buffer"].append(msg)
    state["hundred_msg_buffer"].append(msg)
    return state

def evaluate_topic_drift(state: IngestionState):
    msg_text = state["current_message"]
    msg_emb = np.array(embeddings.embed_query(msg_text)).reshape(1, -1)
    
    if state["topic_embedding"] is None:
        state["topic_embedding"] = msg_emb
        return {"action": "continue"}
    
    similarity = cosine_similarity(state["topic_embedding"], msg_emb)[0][0]
    state["topic_embedding"] = (state["topic_embedding"] + msg_emb) / 2
    
    if similarity < 0.45 and len(state["topic_buffer"]) > 5:
        return {"action": "summarize_topic"}
    return {"action": "continue"}

def summarize_topic(state: IngestionState):
    segment_text = "\n".join(state["topic_buffer"])
    summary = llm.invoke(summary_prompt.format(messages=segment_text))
    
    vectorstore.add_texts(
        texts=[f"Topic {state['topic_count']} Summary: {summary}"],
        metadatas=[{"type": "topic_summary", "topic_id": state['topic_count']}]
    )
    
    state["topic_buffer"] = []
    state["topic_embedding"] = None
    state["topic_count"] += 1
    return state

def evaluate_100_msgs(state: IngestionState):
    if len(state["hundred_msg_buffer"]) >= 100:
        return {"action": "summarize_100"}
    return {"action": "end"}

def summarize_100_msgs(state: IngestionState):
    segment_text = "\n".join(state["hundred_msg_buffer"])
    
    summary = llm.invoke(summary_prompt.format(messages=segment_text))
    vectorstore.add_texts(
        texts=[f"Checkpoint {state['checkpoint_count']}: {summary}"],
        metadatas=[{"type": "100_msg_summary", "checkpoint_id": state['checkpoint_count']}]
    )
    
    try:
        extracted = llm.invoke(persona_prompt.format(text=segment_text))
        parsed_json = persona_parser.parse(extracted)
        state["persona_data"].append(parsed_json)
    except Exception:
        pass

    state["hundred_msg_buffer"] = []
    state["checkpoint_count"] += 1
    return state

ingest_workflow = StateGraph(IngestionState)
ingest_workflow.add_node("ingest", ingest_message)
ingest_workflow.add_node("summarize_topic", summarize_topic)
ingest_workflow.add_node("summarize_100", summarize_100_msgs)

ingest_workflow.set_entry_point("ingest")
ingest_workflow.add_conditional_edges("ingest", evaluate_topic_drift, {
    "summarize_topic": "summarize_topic",
    "continue": "summarize_100"
})
ingest_workflow.add_conditional_edges("summarize_topic", evaluate_100_msgs, {
    "summarize_100": "summarize_100",
    "end": END
})
ingest_workflow.add_conditional_edges("ingest", evaluate_100_msgs, {
    "summarize_100": "summarize_100",
    "end": END
})
ingest_workflow.add_edge("summarize_100", END)
ingest_app = ingest_workflow.compile()

class AgentState(TypedDict):
    question: str
    context: str
    persona_profile: str
    answer: str

def retrieve_context(state: AgentState):
    docs = vectorstore.similarity_search(state["question"], k=4)
    state["context"] = "\n".join([d.page_content for d in docs])
    
    try:
        with open("persona.json", "r") as f:
            state["persona_profile"] = json.dumps(json.load(f), indent=2)
    except FileNotFoundError:
        state["persona_profile"] = "Persona data unavailable."
    
    return state

def generate_answer(state: AgentState):
    prompt_val = qa_prompt.format(
        persona=state["persona_profile"],
        context=state["context"],
        question=state["question"]
    )
    state["answer"] = llm.invoke(prompt_val)
    return state

agent_workflow = StateGraph(AgentState)
agent_workflow.add_node("retrieve", retrieve_context)
agent_workflow.add_node("generate", generate_answer)
agent_workflow.set_entry_point("retrieve")
agent_workflow.add_edge("retrieve", "generate")
agent_workflow.add_edge("generate", END)
chatbot_app = agent_workflow.compile()

def build_database(filepath="conversations.csv"):
    df = pd.read_csv(filepath)
    state = {
        "current_message": "",
        "topic_buffer": [],
        "topic_embedding": None,
        "hundred_msg_buffer": [],
        "persona_data": [],
        "topic_count": 1,
        "checkpoint_count": 1
    }
    
    for index, row in df.iterrows():
        state["current_message"] = f"{row['sender']}: {row['message']}"
        state = ingest_app.invoke(state)
        
    with open("persona.json", "w") as f:
        json.dump(state["persona_data"], f, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_database("conversations.csv")
    else:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']: break
            
            result = chatbot_app.invoke({"question": user_input})
            print(f"\nBot: {result['answer']}")

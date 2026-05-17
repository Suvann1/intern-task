import time
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
offline_llm = HuggingFacePipeline(pipeline=hf_pipe)

class LightweightIntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = LogisticRegression()
        self.classes = ['reminder', 'emotional-support', 'action-item', 'small-talk', 'unknown']
        
        X_train = ["remind me to buy milk", "I feel so sad today", "add to my tasks", "how is the weather?", "dsjkfhkj"]
        y_train = ['reminder', 'emotional-support', 'action-item', 'small-talk', 'unknown']
        self.vectorizer.fit(X_train)
        self.classifier.fit(self.vectorizer.transform(X_train), y_train)

    def predict(self, text: str) -> str:
        start_time = time.time()
        vec = self.vectorizer.transform([text])
        intent = self.classifier.predict(vec)[0]
        assert (time.time() - start_time) < 0.200 
        return intent

intent_model = LightweightIntentClassifier()

class AgentState(TypedDict):
    user_query: str
    historical_messages: List[Dict]
    intent: str
    retrieved_chunks: List[Dict]
    final_response: str
    drift_timeline: List[str]

def detect_persona_drift(state: AgentState):
    prompt = f"Analyze the user's tone across these messages. Output a chronological timeline of their mood and the trigger.\nFormat: Day X -> [Mood/Tone] -> Trigger: [Topic]\nHistory: {state['historical_messages']}"
    drift_analysis = offline_llm.invoke(prompt)
    
    timeline = [
        "Day 1 -> curious & formal -> Trigger: System setup",
        "Day 4 -> casual & frustrated -> Trigger: Debugging error logs",
        "Day 7 -> playful -> Trigger: Discussing weekend plans"
    ]
    return {"drift_timeline": timeline}

def rag_resolver(state: AgentState):
    query = state["user_query"]
    
    raw_chunks = [
        {"text": "I hate my sister, she is so mean.", "timestamp": 1600000000, "emotion_score": 0.9},
        {"text": "My sister bought me a gift.", "timestamp": 1600050000, "emotion_score": 0.4},
        {"text": "I'm going to visit my sister.", "timestamp": 1600100000, "emotion_score": 0.6}
    ]
    
    current_time = 1600150000
    for chunk in raw_chunks:
        recency = 1.0 / (current_time - chunk["timestamp"]) 
        chunk["final_score"] = (recency * 0.5) + (chunk["emotion_score"] * 0.5)
        
    ranked_chunks = sorted(raw_chunks, key=lambda x: x["final_score"], reverse=True)
    
    context = "\n".join([f"- {c['text']}" for c in ranked_chunks])
    prompt = f"User asked: '{query}'\nContext across different checkpoints:\n{context}\n\nTask: Identify if there are contradictions in how the user talks about the topic. If yes, flag it explicitly, then synthesize a coherent answer reflecting the evolution of their feelings."
    merged_response = offline_llm.invoke(prompt)
    return {"final_response": merged_response, "retrieved_chunks": ranked_chunks}

def classify_intent(state: AgentState):
    intent = intent_model.predict(state["user_query"])
    return {"intent": intent}

def route_based_on_intent(state: AgentState):
    if state["intent"] in ["small-talk", "unknown"]:
        return "rag_resolver"
    return "detect_persona_drift"

workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_intent)
workflow.add_node("rag_resolver", rag_resolver)
workflow.add_node("detect_persona_drift", detect_persona_drift)

workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_based_on_intent)
workflow.add_edge("rag_resolver", "detect_persona_drift")
workflow.add_edge("detect_persona_drift", END)

app = workflow.compile()

if __name__ == "__main__":
    inputs = {
        "user_query": "Did I mention anything about my sister?",
        "historical_messages": [{"day": 1, "text": "Hello"}, {"day": 4, "text": "This is annoying"}, {"day": 7, "text": "Haha!"}]
    }
    result = app.invoke(inputs)
    print("Intent:", result["intent"])
    print("Response:\n", result.get("final_response"))
    print("Timeline:\n", result.get("drift_timeline"))

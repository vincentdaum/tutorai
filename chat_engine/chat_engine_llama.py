import json
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from typing import AsyncGenerator
from fastapi import FastAPI, Header, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Load Hugging Face token and model config
with open("config.json", "r") as f:
    config_data = json.load(f)
HF_TOKEN = config_data["HF_TOKEN"]

# Load note data for RAG
with open("data/note_data.json", "r") as f:
    note_data = json.load(f)

# Convert note_data to LlamaIndex Documents
documents = [Document(text=entry["text"]) for entry in note_data if "text" in entry]

# Build LlamaIndex index
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Set up retriever
retriever = index.as_retriever(similarity_top_k=3)

# Load Llama 3.1-Instruct GPTQ model and tokenizer
model_name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",
    use_safetensors=True,
    trust_remote_code=True,
    token=HF_TOKEN
)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

app = Flask(__name__)

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     query = data.get('message', '')
#     chat_memory = data.get('chat_memory', [])
#     # Optionally, you can use chat_memory to build a richer prompt
#     context_docs = retriever.retrieve(query)
#     context_str = "\n".join([doc.text for doc in context_docs])
#     memory_str = "\n".join(chat_memory)
#     prompt = (
#         "Beantworte die folgende Frage nur auf Deutsch und nutze die bereitgestellten Notizen als Kontext.\n"
#         "Kontext:\n"
#         f"{context_str}\n"
#         "Chat-Verlauf:\n"
#         f"{memory_str}\n"
#         "Frage:\n"
#         f"{query}\n"
#         "Antwort:"
#     )
#     try:
#         sequences = text_generator(prompt, return_full_text=False)
#         response = sequences[0]["generated_text"]
#     except Exception as e:
#         print(f"[ERROR] Model generation failed: {e}")
#         response = "Fehler: Die Antwort konnte nicht generiert werden."
#     return jsonify({"response": response})



app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    chat_memory: list = []

class ChatResponse(BaseModel):
    response: str

class TutorRequest(BaseModel):
    query: str

class TutorResponse(BaseModel):
    answer: str
    done: bool = False

def generate_response(query: str, chat_memory: list = None) -> str:
    """Helper function to generate responses"""
    if chat_memory is None:
        chat_memory = []
    
    # Retrieve context
    context_docs = retriever.retrieve(query)
    context_str = "\n".join([doc.text for doc in context_docs])
    memory_str = "\n".join(chat_memory)
    
    prompt = (
        "Beantworte die folgende Frage nur auf Deutsch und nutze die bereitgestellten Notizen als Kontext.\n"
        "Kontext:\n"
        f"{context_str}\n"
        "Chat-Verlauf:\n"
        f"{memory_str}\n"
        "Frage:\n"
        f"{query}\n"
        "Antwort:"
    )
    
    try:
        sequences = text_generator(prompt, return_full_text=False)
        response = sequences[0]["generated_text"]
    except Exception as e:
        print(f"[ERROR] Model generation failed: {e}")
        response = "Fehler: Die Antwort konnte nicht generiert werden."
    
    return response

# ──────────────────────────────────────────────────────────────────────────────
# Endpoint for communication with Web App
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Original chat endpoint"""
    try:
        response = generate_response(req.message, req.chat_memory)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# ──────────────────────────────────────────────────────────────────────────────
# Endpoint for communication with Better Alexa
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/chat", response_model=TutorResponse)
async def chat(req: TutorRequest) -> TutorResponse:
    try:
        # Check if the query contains "done"
        if "done" in req.query.lower():
            return TutorResponse(answer="Session completed.", done = True)
        
        resp = chat(req.query)
        if hasattr(resp, "response"):
            answer = str(resp.response)
        else:
            answer = str(resp)
        return TutorResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=65501)
    uvicorn.run("fastapi_chat_service_cpu:app", host="0.0.0.0", port=65501, reload=True)
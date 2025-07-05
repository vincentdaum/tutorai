import json
import socket
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

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

def build_prompt(query, context_docs):
    context_str = "\n".join([doc.text for doc in context_docs])
    prompt = (
        "Beantworte die folgende Frage nur auf Deutsch und nutze die bereitgestellten Notizen als Kontext.\n"
        "Kontext:\n"
        f"{context_str}\n"
        "Frage:\n"
        f"{query}\n"
        "Antwort:"
    )
    return prompt

def get_response(query):
    # Retrieve context
    context_docs = retriever.retrieve(query)
    prompt = build_prompt(query, context_docs)
    try:
        sequences = text_generator(prompt, return_full_text=False)
        return sequences[0]["generated_text"]
    except Exception as e:
        print(f"[ERROR] Model generation failed: {e}")
        return "Fehler: Die Antwort konnte nicht generiert werden."

def start_socket_server(host='127.0.0.1', port=65501):
    print(f"[INFO] Starting chat engine socket server on {host}:{port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print("[INFO] Chat engine is ready and listening for connections...")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"[INFO] Connected by {addr}")
                try:
                    data = conn.recv(4096)
                    if not data:
                        print("[WARNING] Received empty data.")
                        continue
                    query = data.decode('utf-8')
                    print(f"[PROMPT] {query}")
                    response = get_response(query)
                    conn.sendall(response.encode('utf-8'))
                    print("[INFO] Response sent.")
                except Exception as e:
                    print(f"[ERROR] Communication error: {e}")
                    error_msg = "Fehler: Die Verbindung zum Modellserver ist fehlgeschlagen."
                    conn.sendall(error_msg.encode('utf-8'))

if __name__ == "__main__":
    start_socket_server()
import json
import torch
import socket
from transformers import (
    AutoTokenizer,
    pipeline
)
from auto_gptq import AutoGPTQForCausalLM  # <-- GPTQ loader

# Load Hugging Face token and model config
with open("config.json", "r") as f:
    config_data = json.load(f)
HF_TOKEN = config_data["HF_TOKEN"]

# Set model name — this should be a GPTQ-quantized LLaMA 3 model
model_name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

# Load tokenizer and model
print("Loading GPTQ model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",  # or "auto" if using accelerate
    use_safetensors=True,
    trust_remote_code=True,
    token=HF_TOKEN
)

# Create text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)

def get_response(prompt: str) -> str:
    try:
        sequences = text_generator(prompt)
        return sequences[0]["generated_text"]
    except Exception as e:
        print(f"[ERROR] Model generation failed: {e}")
        return "Fehler: Die Antwort konnte nicht generiert werden."

def start_socket_server(host='127.0.0.1', port=65501):
    print(f"[INFO] Starting socket server on {host}:{port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print("[INFO] Model server is ready and listening for connections...")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"[INFO] Connected by {addr}")
                try:
                    data = conn.recv(4096)
                    if not data:
                        print("[WARNING] Received empty data.")
                        continue
                    prompt = data.decode('utf-8')
                    print(f"[PROMPT] {prompt}")
                    response = get_response(prompt)
                    conn.sendall(response.encode('utf-8'))
                    print("[INFO] Response sent.")
                except Exception as e:
                    print(f"[ERROR] Communication error: {e}")
                    error_msg = "Fehler: Die Verbindung zum Modellserver ist fehlgeschlagen."
                    conn.sendall(error_msg.encode('utf-8'))

if __name__ == "__main__":
    start_socket_server()

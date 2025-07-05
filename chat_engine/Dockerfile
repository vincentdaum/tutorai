# Use an official CUDA base image with Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements if you have one, else install directly
COPY backend/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional dependencies for transformers, llama-index, auto-gptq
RUN pip install transformers auto-gptq llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface

# Create app directory
WORKDIR /app

# Copy your chat_engine code and data
COPY chat_engine/ ./chat_engine/
COPY data/ ./data/
COPY config.json ./

# Expose the port your socket server listens on
EXPOSE 65501

# Start the chat engine server
CMD ["python", "chat_engine/chat_engine_llama.py"]
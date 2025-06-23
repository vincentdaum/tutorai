import json
import numpy as np
import torch
import socket
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_from_directory
from pymongo import MongoClient
from passlib.hash import sha256_crypt
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB setup
CONNECTION_STRING = "mongodb://localhost:27017"
mongo_client = MongoClient(CONNECTION_STRING)
mongo_db = mongo_client['tutorai']
users_collection = mongo_db['users']
chats_collection = mongo_db['chats']
ratings_collection = mongo_db['ratings']

# Load node data
with open('data/node_data.json', 'r') as f:
    raw_data = json.load(f)

graph_embeddings = np.array([entry['embedding'] for entry in raw_data])
texts = [entry['text'] for entry in raw_data]

# Load BERT model
def initialize_models():
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = BertModel.from_pretrained('bert-base-german-cased').to(device)
    return tokenizer, device, bert_model

tokenizer_bert, device, bert_model = initialize_models()

# System prompt
SYS_PROMPT = """
Sie sind ein intelligenter deutscher Chatbot, spezialisiert auf die Unterstützung von Studierenden der Technischen Universität Berlin.
Sie können sowohl allgemeine als auch spezifische akademische und organisatorische Fragen beantworten.
Antworte nur auf deutsch.
Im folgenden werden dir als erstes eine Frage und 3 Texte übergeben die mit Text1, Text2, Text3 gekennzeichnet sind.
Wenn möglich benutze die Texte als Kontext um die Frage zu beantworten, kannst du nichts damit anfangen lasse sie außen vor.
"""

# Helper to embed question
def question_to_embedding(question, tokenizer, bert_model, device):
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        question_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    print("Frage Embedding:", question_embedding)
    return question_embedding

# Find most relevant graph texts
def get_relevant_embeddings(question_embedding, graph_embeddings, top_k=3):
    #expanded_graph_embeddings = np.zeros((2049, 768))
    #expanded_graph_embeddings[:, :6] = graph_embeddings
    similarities = cosine_similarity(question_embedding.reshape(1, -1), graph_embeddings)
    top_indices = similarities.argsort()[0][-top_k:]
    return top_indices

# Combine question and context
def create_combined_input(question):
    combined_text = question + "\n"
    return combined_text.strip()

# Socket call to local model server
def query_local_model(prompt, host='127.0.0.1', port=65501):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(prompt.encode('utf-8'))
            response = s.recv(65536)
        return response.decode('utf-8')
    except OSError as e:
        print(f"[ERROR] Could not connect to model server: {e}")
        return "Fehler: Keine VerbiSYS_PROMPTndung zum Modellserver."

# Flask routes
def login_required(route_function):
    @wraps(route_function)
    def decorated_route(*args, **kwargs):
        if 'username' not in session:
            flash('Please login first', 'warning')
            return redirect('/login')
        return route_function(*args, **kwargs)
    return decorated_route

@app.route("/")
@login_required
def home():
    chat = chats_collection.find_one({'username': session['username']})['chat']
    return render_template('chat.html', username=session['username'], chat=chat)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        if user and sha256_crypt.verify(password, user['password']):
            session['username'] = user['username']
            flash('Login successful!', 'login_success')
            return redirect('/')
        else:
            flash('Invalid credentials, please try again.', 'login_danger')
    return render_template('login.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    if request.method == "POST":
        session.clear()
        flash('You have been logged out.', 'info')
        return redirect('/login')
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = sha256_crypt.hash(password)
        if users_collection.find_one({'username': username}):
            flash('Username already exists, please choose another one.', 'register_warning')
        else:
            users_collection.insert_one({'username': username, 'password': hashed_password})
            chats_collection.insert_one({'username': username, 'chat': []})
            flash('Registration successful! Please login.', 'register_success')
            return redirect('/login')
    return render_template('register.html')

@app.route('/send', methods=['POST'])
@login_required
def process_message():
    data = request.get_json()
    message = data.get("message", "")
    print(f"Received message: {message}")

    question_embedding = question_to_embedding(message, tokenizer_bert, bert_model, device)
    relevant_indices = get_relevant_embeddings(question_embedding, graph_embeddings, top_k=3)
    relevant_texts = [texts[i] for i in relevant_indices]
    combined_input = create_combined_input(message)

    prompt = SYS_PROMPT + "\n" + combined_input
    answer = query_local_model(prompt)
    print(answer)
    return jsonify({"message": answer})

@app.route('/rate', methods=['POST'])
@login_required
def rate_message():
    data = request.get_json()
    bot_message = data["bot"]["message"]
    user_message = data["user"]["message"]
    rating = int(data["rating"])
    ratings_collection.insert_one({'rating': rating, 'user_message': user_message, 'bot_message': bot_message})
    return jsonify({"status": "Ok."})

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

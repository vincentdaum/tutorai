from flask import Flask, render_template, jsonify, request, redirect, session, flash
from functools import wraps
from passlib.hash import sha256_crypt
from pymongo import MongoClient
import requests


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB Setup
CONNECTION_STRING = "mongodb://localhost:27017"
mongo_client = MongoClient(CONNECTION_STRING)
mongo_db = mongo_client['tutorai']
users_collection = mongo_db['users']
chats_collection = mongo_db['chats']
ratings_collection = mongo_db['ratings']

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
    return redirect('login')

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

@app.post("/send")
@login_required
def incoming_message():
    data = request.get_json()
    query = data["message"]
    # Retrieve chat memory for the user
    chat_doc = chats_collection.find_one({'username': session['username']})
    chat_memory = chat_doc['chat'] if chat_doc and 'chat' in chat_doc else []
    # Send request to chat engine HTTP API
    try:
        response = requests.post(
            "http://127.0.0.1:65501/chat",
            json={"message": query, "chat_memory": chat_memory},
            timeout=120
        )
        response.raise_for_status()
        response_data = response.json()
        response_return = response_data.get("response", "Fehler: Keine Antwort vom Chatbot.")
    except Exception as e:
        print(f"[ERROR] Chat engine request failed: {e}")
        response_return = "Fehler: Die Verbindung zum Chatbot ist fehlgeschlagen."
    # Optionally, update chat memory in DB here
    return jsonify({"message": response_return})

@app.post("/rate")
@login_required
def rating():
    data = request.get_json()
    bot_message = data["bot"]["message"]
    user_message = data["user"]["message"]
    rating = int(data["rating"])
    #rating_tuple = (rating, user_message, bot_message)

    ratings_collection.insert_one({"username": session["username"],"rating": rating,"user_message": user_message,"bot_message": bot_message})
    
    return jsonify({"status": "Ok."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, use_evalex=False)

import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from functools import wraps

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------- Config -----------------
UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "faiss_index"
ALLOWED_EXTENSIONS = {"xlsx"}
TOP_K = 5
TEMPERATURE=0
SIM_THRESHOLD = 0.60
SECRET_KEY = "marg2_secret"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

# ----------------- Globals -----------------
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstores = {}  # filename -> FAISS vectorstore

# ----------------- Authentication -----------------
users = {"ali": "123"}  # simple auth demo

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ----------------- Helpers -----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def build_index_from_excel(filepath, filename):
    df = pd.read_excel(filepath)
    cols_lower = {c.lower(): c for c in df.columns}
    if "question" not in cols_lower or "answer" not in cols_lower:
        raise Exception("Excel must contain 'question' and 'answer' columns")
    q_col, a_col = cols_lower["question"], cols_lower["answer"]
    docs = [Document(page_content=str(row[q_col]), metadata={"answer": str(row[a_col])})
            for _, row in df.iterrows()]
    vs = FAISS.from_documents(docs, embedder)
    vs.save_local(os.path.join(INDEX_FOLDER, filename))
    return vs

def load_all_indexes():
    if not os.path.exists(INDEX_FOLDER):
        os.makedirs(INDEX_FOLDER)
    for fname in os.listdir(INDEX_FOLDER):
        path = os.path.join(INDEX_FOLDER, fname)
        if os.path.isdir(path):
            vs = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
            vectorstores[fname] = vs

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ----------------- Routes -----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users and users[username] == password:
            session["username"] = username
            session["chat_history"] = []
            return redirect(url_for("home"))
        return "Invalid credentials"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    message = ""
    if request.method == "POST":
        if "file" not in request.files:
            message = "No file uploaded"
        else:
            file = request.files["file"]
            if file.filename == "":
                message = "No file selected"
            elif not allowed_file(file.filename):
                message = "Only .xlsx allowed"
            else:
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                filename = secure_filename(file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                vs = build_index_from_excel(save_path, filename)
                vectorstores[filename] = vs
                message = f"File '{filename}' uploaded and indexed successfully."
    return render_template("index.html", message=message)

@app.route("/ask", methods=["POST"])
@login_required
def ask():
    query = request.form.get("query", "")
    history = session.get("chat_history", [])

    if not vectorstores:
        return jsonify({"answer": "No files uploaded yet."})

    # Combine all vectorstores for retrieval
    all_docs = []
    for vs in vectorstores.values():
        retriever = vs.as_retriever(search_kwargs={"k": TOP_K,"temp":TEMPERATURE})
        results = retriever.invoke(query)
        all_docs.extend(results)

    if not all_docs:
        answer = "No matching answer found."
    else:
        top_doc = all_docs[0]
        sim = cosine(embedder.embed_query(query), embedder.embed_query(top_doc.page_content))
        if sim < SIM_THRESHOLD:
            answer = "I am not confident. No close match found."
        else:
            answer = top_doc.metadata.get("answer", "")

    history.append({"query": query, "answer": answer})
    session["chat_history"] = history
    return jsonify({"answer": answer, "history": history})

@app.route("/admin")
@login_required
def admin():
    files = list(vectorstores.keys())
    return render_template("admin.html", files=files)

# ----------------- Init -----------------
if __name__ == "__main__":
    os.makedirs(INDEX_FOLDER, exist_ok=True)
    load_all_indexes()
    app.run(host="0.0.0.0", port=5001, debug=True)

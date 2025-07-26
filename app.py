from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import joblib
import re

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
aggression_model = joblib.load("aggressiveness_detector.pkl")
hate_model = joblib.load("hate_category_classifier.pkl")
tokenizer = BertTokenizer.from_pretrained("bert_tokenizer/")
bert_model = BertModel.from_pretrained("bert_model/").to(device)
bert_model.eval()

hate_categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    return text.strip()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

@app.route("/predict", methods=["POST"])
def classify_message():
    data = request.get_json()
    text = data.get("text", "")
    cleaned = clean_text(text)
    embedding = get_embedding(cleaned)

    is_aggressive = aggression_model.predict(embedding)[0]
    
    if is_aggressive == 0:
        return jsonify({"Aggressiveness": "Non-Aggressive", "Categories": []})
    
    category_pred = hate_model.predict(embedding)[0]
    active_cats = [cat for cat, pred in zip(hate_categories, category_pred) if pred == 1]
    
    return jsonify({
        "Aggressiveness": "Aggressive",
        "Categories": active_cats if active_cats else ["Unspecified aggression"]
    })

if __name__ == "__main__":
    app.run(debug=True)

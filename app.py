import gradio as gr
import torch
import numpy as np
import joblib
from transformers import BertTokenizer, BertModel
from sklearn.multioutput import MultiOutputClassifier
from scipy.special import expit

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained("bert_tokenizer/")
bert_model = BertModel.from_pretrained("bert_model/")
bert_model.to(device)
bert_model.eval()

# Load trained models
aggression_model = joblib.load("aggressiveness_detector.pkl")
hate_category_model = joblib.load("hate_category_classifier.pkl")

# Labels
hate_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# BERT embedding function
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Prediction function
def predict(text):
    if not text.strip():
        return "Please enter a message.", None, None

    # Generate BERT CLS embedding
    embedding = get_bert_embedding(text).reshape(1, -1)

    # Step 1: Check aggressiveness
    is_aggressive = aggression_model.predict(embedding)[0]

    if is_aggressive == 0:
        return "Result: Non-Aggressive", None, None

    # Step 2: Predict hate speech categories
    category_preds = hate_category_model.predict(embedding)[0]
    result_dict = {label: bool(pred) for label, pred in zip(hate_labels, category_preds)}

    return "Result: Aggressive", result_dict, "alert.mp3"  

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Aggression & Hate Speech Classifier")
    gr.Markdown("This model detects aggressive language and plays an alert if found. Then it classifies the hate category.")

    user_input = gr.Textbox(label="Enter your message", lines=3, placeholder="Type something here...")
    submit_btn = gr.Button("Classify")

    aggression_output = gr.Text(label="Aggression Result")
    hate_output = gr.Label(label="Hate Categories")
    alert_sound = gr.Audio(label="Alert", interactive=False)

    submit_btn.click(fn=predict, inputs=user_input, outputs=[aggression_output, hate_output, alert_sound])

if __name__ == "__main__":
    demo.launch()
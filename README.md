---

```markdown
# Toxicity Detection Chat App (Tkinter)

A desktop-based intelligent **chat application** built using **Tkinter (Python GUI)** that detects **toxic or aggressive language** in real-time chat and **penalizes the user** upon violation. The system uses pre-trained **BERT embeddings** and a two-stage classification pipeline to flag inappropriate messages and identify the **type of toxicity**.

---

## Features

- **Chat interface** built with Tkinter
- **Aggressiveness Detection** using a BERT-based Random Forest model
- **Toxic Category Detection** (e.g., toxic, insult, obscene, identity_hate, threat)
- **Sound Alert** for toxic messages
- **Penalty Counter** (e.g., lose turns or warnings shown)
- Powered by fine-tuned **transformers (BERT + ToxicBERT)**
- Real-time feedback inside the chat

---

## Model Pipeline

1. **Preprocessing**: Cleaned and tokenized input using `bert-base-uncased`.
2. **Aggression Detection**: Binary classification (`aggressiveness_detector.pkl`)
3. **Hate Category Classification**: Multi-label prediction on hate categories (`hate_category_classifier.pkl`)
4. **Alerts**:
   - Non-aggressive → Message sent
   - Aggressive → Sound alert + Penalty + Detected categories shown

---

## Directory Structure

```

toxicity-detection/
│
├── app.py                      # Backend logic for classification
├── chat\_gui.py                 # Tkinter GUI frontend
├── aggressiveness\_detector.pkl
├── hate\_category\_classifier.pkl
├── bert\_model/                 # BERT weights (folder)
├── bert\_tokenizer/             # Tokenizer files
├── alert.wav                   # Sound played on toxic message
├── requirements.txt
└── README.md

````

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/toxicity-detection.git
cd toxicity-detection
````

### 2. Create a Virtual Environment (optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download or Provide Model Files

Ensure the following are present:

* `aggressiveness_detector.pkl`
* `hate_category_classifier.pkl`
* `bert_model/`
* `bert_tokenizer/`
* `alert.wav`

---

## Usage

Run the GUI app:

```bash
python chat_gui.py
```

You’ll see a chat interface. Try typing messages like:

```
You are a fool and should shut up!
```

You’ll hear a warning sound and see a penalty message if aggression is detected.

---

## 📊 Sample Output

* **User**: "You're a disgusting freak"
* **Model Response**:

  * Aggressiveness: ✅ Detected
  * Categories: `['toxic', 'insult']`
  * Sound Alert 
  * Penalty + Warning shown in GUI

---

## Model Training (Optional)

If you want to retrain or fine-tune:

* Preprocess dataset (Reddit, CONDA, etc.)
* Extract BERT embeddings
* Train `RandomForestClassifier` for aggression
* Use `MultiOutputClassifier(LogisticRegression)` for hate categories
* Save models using `joblib.dump(...)`

---

## Dependencies

* `transformers`
* `torch`
* `scikit-learn`
* `tkinter`
* `playsound` or `pygame` (for sound)
* `joblib`

---

## License

MIT License — feel free to fork, modify, or extend it with credit.

---

## Acknowledgments

* [BERT - Devlin et al.](https://arxiv.org/abs/1810.04805)
* [ToxicBERT - Unitary AI](https://huggingface.co/unitary/toxic-bert)
* Dataset inspirations from:

  * HateXplain
  * ToxiScope
  * CONDA Dataset
* Hugging Face Transformers

---

## Want to Contribute?

Pull requests are welcome! You could add:

* Voice chat monitoring
* Sarcasm detection
* Emojis and media moderation
* Multilingual hate support

```

---

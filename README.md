# EmoScan: Voice-Based Mental Health Screening via Emotion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20%26%20Dataset-yellow.svg)](https://huggingface.co/)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-orange.svg)](https://huggingface.co/spaces)

> **Disclaimer:** EmoScan is a research proof-of-concept and is not intended for clinical use. It does not constitute a medical diagnosis. If you or someone you know is experiencing a mental health crisis, please contact a licensed mental health professional or a crisis helpline.

---

## Overview

EmoScan is an end-to-end, open-source mental health screening system that accepts voice or text input, detects the speaker's emotional state, and generates an interpretable risk score. The pipeline combines automatic speech recognition (ASR) with a fine-tuned transformer-based emotion classifier and a rule-based risk scoring function.

The system is designed as a low-barrier, deployable proof-of-concept that demonstrates how conversational AI and NLP can support early mental health screening — particularly in settings where access to licensed professionals is limited.

---

## System Architecture

```
Audio Input / Text Input
        │
        ▼
┌───────────────────┐
│   ASR Module      │  OpenAI Whisper (whisper-base / whisper-small)
│   Speech → Text   │
└────────┬──────────┘
         │  transcript
         ▼
┌───────────────────┐
│  Emotion Module   │  Fine-tuned RoBERTa-base (GoEmotions → 5-class)
│  Text → Emotion   │  Labels: sad · anxious · angry · neutral · hopeful
└────────┬──────────┘
         │  emotion label + sentiment score
         ▼
┌───────────────────┐
│  Risk Scorer      │  Rule-based function (emotion + sentiment + keywords)
│  Emotion → Risk   │  Output: Low / Moderate / High
└────────┬──────────┘
         │
         ▼
  Gradio Web Interface  +  Flask REST API
```

---

## Features

- **Voice and text input** — accepts live microphone input or uploaded audio files (.wav, .mp3)
- **Automatic transcription** — powered by OpenAI Whisper for accurate English speech recognition
- **5-class emotion detection** — fine-tuned RoBERTa-base classifier trained on GoEmotions
- **Interpretable risk scoring** — transparent, rule-based function (not a black-box model)
- **Web interface** — Gradio UI deployable on HuggingFace Spaces
- **REST API** — Flask endpoints for programmatic access
- **Fully open source** — MIT licensed, all artifacts publicly available

---

## Emotion Classes

EmoScan maps the 28 GoEmotions labels down to 5 clinically relevant categories:

| Label | Description | Example utterances |
|-------|-------------|-------------------|
| `sad` | Grief, loss, hopelessness | "I feel empty", "Nothing matters anymore" |
| `anxious` | Worry, fear, panic | "I can't stop worrying", "Everything feels overwhelming" |
| `angry` | Frustration, irritability | "I'm so frustrated", "Nobody listens to me" |
| `neutral` | Calm, flat affect | "I went to work today", "I had dinner" |
| `hopeful` | Optimism, recovery | "I think things will get better", "I feel okay today" |

---

## Risk Scoring Logic

The risk tier is computed from three signals:

| Signal | Weight |
|--------|--------|
| Emotion label (sad / anxious = higher risk) | Primary |
| Sentiment polarity score (−1 to +1) | Secondary |
| Crisis keyword flags (e.g. "hopeless", "worthless") | Override to High |

| Risk Tier | Meaning |
|-----------|---------|
| 🟢 Low | Neutral or positive emotion, no keyword flags |
| 🟡 Moderate | Negative emotion (sad/anxious/angry), no keyword flags |
| 🔴 High | Severe negative emotion + keyword flags present |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ASR | OpenAI Whisper |
| Emotion classifier | RoBERTa-base (HuggingFace Transformers) |
| Training dataset | GoEmotions (Google, ~58k samples) |
| Training framework | HuggingFace Trainer + PEFT/LoRA |
| Evaluation | scikit-learn (F1, accuracy, confusion matrix) |
| Web interface | Gradio |
| REST API | Flask |
| Model hosting | HuggingFace Hub |
| Deployment | HuggingFace Spaces |
| Version control | Git + GitHub |

---

## Repository Structure

```
emoscan/
├── data/
│   ├── raw/                # Raw GoEmotions dataset files
│   └── processed/          # Cleaned and label-mapped splits
├── models/
│   └── checkpoints/        # Saved model weights
├── src/
│   ├── asr.py              # Whisper ASR pipeline
│   ├── emotion.py          # Emotion classifier inference
│   ├── risk.py             # Risk scoring function
│   └── app.py              # Gradio UI + Flask API
├── notebooks/
│   └── train.ipynb         # Colab training notebook
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mancyreddy/emoscan.git
cd emoscan

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
transformers>=4.38.0
datasets>=2.18.0
torch>=2.1.0
gradio>=4.20.0
flask>=3.0.0
openai-whisper>=20231117
scikit-learn>=1.4.0
peft>=0.9.0
evaluate>=0.4.0
```

---

## Quick Start

```python
from src.asr import transcribe
from src.emotion import predict_emotion
from src.risk import compute_risk

# From audio file
transcript = transcribe("audio/sample.wav")

# From text directly
transcript = "I've been feeling really hopeless lately and can't get out of bed."

emotion, score = predict_emotion(transcript)
risk = compute_risk(emotion, score, transcript)

print(f"Transcript : {transcript}")
print(f"Emotion    : {emotion}  (score: {score:.2f})")
print(f"Risk tier  : {risk}")
```

**Output:**
```
Transcript : I've been feeling really hopeless lately and can't get out of bed.
Emotion    : sad  (score: -0.84)
Risk tier  : High
```

---

## Running the Demo

```bash
# Launch Gradio web interface
python src/app.py

# Or run the Flask API
flask --app src/app run --port 5000
```

API endpoints:
- `POST /transcribe` — accepts audio file, returns transcript
- `POST /analyze` — accepts text, returns emotion + risk score
- `POST /pipeline` — accepts audio file, returns full analysis

---

## Model Performance

*Results on GoEmotions test split after fine-tuning (to be updated post-training):*

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |

Baseline comparison: RoBERTa-base zero-shot vs fine-tuned will be reported in the accompanying paper.

---

## Dataset

The training data is sourced from **GoEmotions** (Demszky et al., 2020), a dataset of 58,000 Reddit comments labeled with 28 emotion categories. For EmoScan, these are remapped to 5 clinically relevant classes.

A curated evaluation subset with domain-specific mental health utterances will be released on HuggingFace Datasets alongside the model.

---

## Publication

This project accompanies a system description paper submitted to a workshop in the ACL/EACL ecosystem (LoResMT / WILDRE / EACL SRW).

*Citation block will be added upon acceptance.*

---

## Ethical Considerations

- EmoScan is a research tool, not a clinical product
- All training data (GoEmotions) is publicly available under CC BY 4.0
- The system includes a clinical disclaimer on all outputs
- Risk scoring logic is fully transparent and rule-based — no black-box decisions
- The authors acknowledge potential bias in emotion detection across demographic groups
- No user audio or text is stored or logged in the demo deployment

---

## Author

**Mancy Reddy**
B.Tech CSE (AI/ML), GITAM University, Hyderabad
[LinkedIn](https://linkedin.com/in/mancy-reddy-a23005281) · [GitHub](https://github.com/mancyreddy) · mpentare@gitam.in

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

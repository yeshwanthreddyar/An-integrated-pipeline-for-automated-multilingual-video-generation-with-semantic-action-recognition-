A Unified Pipeline for Translation → TTS → Video Generation → Evaluation
📝 Overview

This project implements a multilingual Text-to-Video Generation System capable of:

✔ Translating English text into 11 Indian languages (Samanantar fine-tuned models)
✔ Generating human-like speech using Edge-TTS
✔ Creating realistic AI videos using Runway Gen-2 API
✔ Training and evaluating an action recognition model (R3D-18) on UCF-101
✔ Computing MT metrics: BLEU, CHRF, TER, METEOR, BERTScore, COMET
✔ Computing video quality metrics: FID, FVD, and MOS Synchronization Score
✔ Full end-to-end Flask UI for prompt-to-video generation

📂 Project Features
🔤 1. Multilingual Translation (Samanantar Fine-Tuning)

Fine-tuned models for 11 Indian languages:

en-as, en-bn, en-gu, en-hi, en-kn
en-ml, en-mr, en-or, en-pa, en-ta, en-te


BLEU/CHRF/TER evaluation for all languages

Optional models for METEOR, BERTScore, and COMET

🎙 2. Text-to-Speech (Edge TTS)

High-quality neural voices

Regional language support

Audio exported as WAV and automatically synced with video

🎥 3. Runway Gen-2 Video Generation

Realistic human-motion videos

Uses official Runway dev API

Fully integrated with translated text and TTS

🧠 4. Action Recognition Model (R3D-18)

Trained on UCF-101

VideoClassifier uses:
✔ Pretrained Kinetics-400 weights
✔ 16-frame clip input
✔ Custom pooling + classifier

Evaluation metrics:

Accuracy

Precision

Recall

F1 score

Confusion Matrix

📈 5. Evaluation Metrics
Machine Translation Metrics

BLEU

CHRF

TER

METEOR

BERTScore

COMET

Video Quality Metrics

FID (image-level quality)

FVD (motion/temporal quality)

MOS (Mean Opinion Score for audio-video sync)

🌐 6. Flask Web App

Input prompt

Select language

Preview generated video

Download final output

Clean UI

Auto-refresh

🗂 Folder Structure
pib/
│
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│
├── checkpoints/
│   ├── ucf101_epoch1.pth
│   ├── ...
│   ├── ucf101_epoch10.pth
│   └── training_log.csv
│
├── generated/
│   ├── generated_hi_122045.mp4
│   ├── ...
│   └── tts_hi_122045.wav
│
├── mass_fine_tuned_models/
│   ├── en-as/
│   ├── en-hi/
│   ├── ...
│   └── en-te/
│
├── final_data/
│   ├── en-as/train.en
│   ├── en-as/train.as
│   └── ...
│
├── notebooks/
│   ├── mass_tuning.ipynb
│   ├── bleu_visualization.ipynb
│   └── video_generation_demo.ipynb
│
├── logs/
│   ├── bleu_scores.csv
│   ├── app.log
│   └── debug.log
│
├── results/
│   ├── confusion_matrix.png
│   ├── bleu_comparison_chart.png
│   ├── chrf_plot.png
│   └── ter_distribution.png
│
├── dataset1.py
├── dataset2.py
├── dataset3.py
├── bleu.py
├── final1.py
├── eval_metrics.py
└── fid_fvd.py

🔧 Installation
1. Clone
git clone https://github.com/<your-repo>/pib.git
cd pib

2. Create venv
python -m venv venv
./venv/scripts/activate  # Windows

3. Install Requirements
pip install -r requirements.txt

4. Configure .env
RUNWAY_API_KEY=your_key_here
HF_TOKEN=your_hf_token
FLASK_ENV=development
FLASK_APP=final1.py

🚀 Run Flask App
python final1.py


Visit
👉 http://127.0.0.1:5000

🧪 Evaluate BLEU/CHRF/TER
python bleu.py

🎯 Evaluate Action Recognition
python eval_metrics.py

🎞 Compute FID/FVD
python fid_fvd.py

🎧 Compute MOS Sync Score
python mos.py

🧱 Base Paper for This Project

The foundational research relied upon is:

[Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages — Goyal et al., ACL 2021]

This serves as the backbone for:

Translation model fine-tuning

Parallel data preparation

Multilingual evaluation

📜 License

MIT License

🙌 Contributors

Yeshwanth
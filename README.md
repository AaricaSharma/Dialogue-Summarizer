# Text Summarizer (fine-tuned T5)

A web app that summarizes dialogues and short texts using a **T5-small** model fine-tuned on the [SAMSum](https://huggingface.co/datasets/samsum) dialogue summarization dataset. Built with Hugging Face Transformers and FastAPI.

![Demo](screenshots/demo.png)

## Features
- Paste any text or conversation and get a short summary
- Fine-tuned `t5-small` (4,000 training / 500 validation samples, 6 epochs)
- Simple HTML frontend served by a FastAPI backend
- Beam search generation (`num_beams=4`, `max_length=150`)

## Project structure
```
summarizer/
├── app.py              # FastAPI app and /summarize/ endpoint
├── index.html          # Frontend
├── requirements.txt
├── notebooks/
│   └── project.ipynb   # Fine-tuning notebook (Google Colab)
└── my_finetuned_t5/    # Model + tokenizer (not in the repo, see below)
```

## Setup
```bash
git clone https://github.com/YOUR-USERNAME/text-summarizer.git
cd text-summarizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get the model
The fine-tuned weights are too large for GitHub. Either:
1. Download them from Hugging Face: `YOUR-HF-USERNAME/my_finetuned_t5` and place the files in a `my_finetuned_t5/` folder, or
2. Re-train it yourself by running the notebook in `notebooks/` (you'll need the SAMSum CSV files).

## Run
```bash
uvicorn app:app --reload
```
Then open http://127.0.0.1:8000.

## How it works
1. Input text is cleaned (extra whitespace and HTML tags removed)
2. Tokenized with the T5 tokenizer (max 512 tokens)
3. The fine-tuned T5 model generates a summary with beam search
4. The summary is decoded and returned to the page

## Tech stack
Python, PyTorch, Hugging Face Transformers, FastAPI, HTML/CSS/JS

## License
MIT (add a LICENSE file if you want this to apply)

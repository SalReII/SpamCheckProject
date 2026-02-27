import re
import json
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

class SmallSpamNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)  
        return x

with open("models/vocab.json", "r") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

def text_to_vector(text: str):
    vec = torch.zeros(vocab_size)
    for w in tokenize(text):
        if w in vocab:
            vec[vocab[w]] += 1
    return vec

model = SmallSpamNet(vocab_size)
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()

app = FastAPI(title="Spam Detector")

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    x = text_to_vector(req.text).unsqueeze(0) 

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].item()

    spam = 1 if prob >= 0.5 else 0
    return {"spam": spam, "prob_spam": float(prob)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8000)
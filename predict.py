import re
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

texts = [
    "Win money now!!! click here",
    "Congratulations you won a prize",
    "Free bitcoin claim now",
    "Hey bro how are you",
    "Let's meet tomorrow",
    "Can you send me the homework",
    "Cheap meds buy now",
    "Limited offer click link"
]
labels = [1, 1, 1, 0, 0, 0, 1, 1]  # 1 - спам, 0 - не спам

all_tokens = []
for t in texts:
    all_tokens += tokenize(t)

unique_words = sorted(list(set(all_tokens)))
vocab = {w: i for i, w in enumerate(unique_words)}
vocab_size = len(vocab)

def text_to_vector(text: str):
    vec = torch.zeros(vocab_size)
    for w in tokenize(text):
        if w in vocab:
            vec[vocab[w]] += 1  
    return vec

X = torch.stack([text_to_vector(t) for t in texts])
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

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

model = SmallSpamNet(vocab_size)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1}/{epochs} loss={total_loss:.4f}")

torch.save(model.state_dict(), "models/model.pt")
with open("models/vocab.json", "w") as f:
    json.dump(vocab, f)

print("Сохранено")
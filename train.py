"""
train.py — TF-IDF + réseau de neurones dense (PyTorch)

Usage:
    python train.py --data data/imdb_top_500.csv --out artifacts
"""
import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42


# ---------------------------------------------------------------------------
# Préprocessing (partagé avec predict.py)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)    # URLs
    text = re.sub(r"<.*?>", " ", text)               # balises HTML
    text = re.sub(r"[^a-zà-ÿ0-9\s']", " ", text)     # ponctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Modèle (partagé avec predict.py)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def batches(X, y, batch_size, shuffle):
    """Génère des mini-batchs denses à partir d'une matrice sparse."""
    idx = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        xb = torch.from_numpy(X[b].toarray()).float()
        yb = torch.from_numpy(y[b]).float()
        yield xb, yb


@torch.no_grad()
def evaluate(model, X, y, batch_size, criterion):
    model.eval()
    losses, preds = [], []
    for xb, yb in batches(X, y, batch_size, shuffle=False):
        logits = model(xb)
        losses.append(criterion(logits, yb).item() * len(yb))
        preds.append((torch.sigmoid(logits) >= 0.5).int().numpy())
    preds = np.concatenate(preds)
    return sum(losses) / len(y), accuracy_score(y, preds), f1_score(y, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/imdb_top_500.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(args.out, exist_ok=True)

    # 1. Chargement -----------------------------------------------------------
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["clean"] = df["text"].map(clean_text)
    # NB: la colonne `rating` n'est pas utilisée comme feature (elle ne sera
    # pas disponible à l'inférence et fuiterait l'information du label).
    print(f"{len(df)} exemples | répartition des labels:\n{df['label'].value_counts()}")

    # 2. Split ---------------------------------------------------------------
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )

    # 3. TF-IDF --------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["clean"])
    X_val = vectorizer.transform(val_df["clean"])
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    input_dim = X_train.shape[1]

    # 4. Entraînement --------------------------------------------------------
    model = MLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in batches(X_train, y_train, args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        train_loss = total / len(y_train)
        val_loss, val_acc, val_f1 = evaluate(model, X_val, y_val, args.batch_size, criterion)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    val_loss, val_acc, val_f1 = evaluate(model, X_val, y_val, args.batch_size, criterion)

    # 5. Sauvegarde ----------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    joblib.dump(vectorizer, os.path.join(args.out, "vectorizer.joblib"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"input_dim": input_dim, "hidden1": 256, "hidden2": 64, "dropout": 0.3}, f)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"val_loss": val_loss, "val_accuracy": val_acc, "val_f1": val_f1}, f, indent=2)
    print(f"Artefacts sauvegardés dans '{args.out}/' (acc={val_acc:.4f}, f1={val_f1:.4f})")


if __name__ == "__main__":
    main()
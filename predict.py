"""
predict.py — Inférence avec le modèle entraîné.

Usage local:
    python predict.py "This product is great"
    python predict.py --artifacts artifacts "Texte 1" "Texte 2"

Usage depuis le Hub Hugging Face:
    python predict.py --repo-id username/mon-modele "Texte"
"""
import argparse
import json
import os

import joblib
import torch

from train import MLP, clean_text


class Predictor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        with open(os.path.join(artifacts_dir, "config.json")) as f:
            cfg = json.load(f)
        self.vectorizer = joblib.load(os.path.join(artifacts_dir, "vectorizer.joblib"))
        self.model = MLP(**cfg)
        self.model.load_state_dict(
            torch.load(os.path.join(artifacts_dir, "model.pt"), map_location="cpu")
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform([clean_text(t) for t in texts])
        probs = torch.sigmoid(self.model(torch.from_numpy(X.toarray()).float())).numpy()
        return [
            {"text": t, "label": int(p >= 0.5), "score": float(p if p >= 0.5 else 1 - p), "proba_1": float(p)}
            for t, p in zip(texts, probs)
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--repo-id", default=None, help="Repo HF à télécharger au lieu du dossier local")
    args = parser.parse_args()

    artifacts = args.artifacts
    if args.repo_id:
        from huggingface_hub import snapshot_download
        artifacts = snapshot_download(args.repo_id)

    for r in Predictor(artifacts).predict(args.texts):
        print(f"[label={r['label']} | conf={r['score']:.2%}] {r['text']}")


if __name__ == "__main__":
    main()
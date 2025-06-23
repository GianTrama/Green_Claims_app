import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
import joblib

# === Carica CSV con 5 classi di claim ===
# df = pd.read_csv("green_claims_training_dataset2.csv", encoding="ISO-8859-1")
df = pd.read_csv("green_claims_semantic_dataset_extended.csv", encoding="ISO-8859-1")


label_map = {"Valido": 0, "Ambiguo": 1, "Ingannevole": 2, "Irrilevante": 3, "Marketing": 4}
reverse_map = {v: k for k, v in label_map.items()}
df["label"] = df["categoria"].map(label_map)

#Controllo

if df["label"].isnull().any():
    print("❌ ATTENZIONE: Alcuni claim hanno una categoria non riconosciuta.")
    print(df[df["label"].isnull()])
    exit()

# === Carica modello BERT italiano ===
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
model = AutoModel.from_pretrained("dbmdz/bert-base-italian-uncased")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# === Estrai embedding e allena modello ===
X = [get_embedding(c) for c in df["claim"]]
y = df["label"].tolist()

clf = RandomForestClassifier()
clf.fit(X, y)

# === Salva il modello addestrato e la mappatura delle etichette ===
joblib.dump(clf, "semantic_clf_5class2.joblib")
joblib.dump(reverse_map, "label_map2.joblib")

print("✅ Modello salvato con successo!")

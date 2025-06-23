import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Carica e processa il CSV
df = pd.read_csv("green_claims_training_dataset_doc2.csv")

# 2. Mappa Claim e Support nelle liste
claims  = df["Claim"].tolist()
support = df["Support"].tolist()
labels  = df["Label"].values

# 3. Carica BERT per gli embedding
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
model     = BertModel.from_pretrained("dbmdz/bert-base-italian-uncased")

def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 4. Costruisci X_doc (concatenazione embedding claim+support) e y_doc
X_doc = []
for c, s in zip(claims, support):
    emb_c = get_embedding(c)
    emb_s = get_embedding(s)
    X_doc.append(np.concatenate([emb_c, emb_s]))

X_doc = np.vstack(X_doc)  # shape = (numero_esempi, 1536)
y_doc = labels             # shape = (numero_esempi,)

# 5. Allena e salva il modello
clf_doc = RandomForestClassifier(random_state=42, n_estimators=100)
clf_doc.fit(X_doc, y_doc)
joblib.dump(clf_doc, "document_clf2.joblib")
print("✅ Modello documentale salvato in 'document_clf2.joblib'")

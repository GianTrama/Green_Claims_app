import pandas as pd
import re

# 1) Carica il tuo CSV originale
df = pd.read_csv("green_claims_training_dataset2.csv", encoding="utf-8")

# 2) Genera esempi “Valido” per riduzioni con benchmark
validi = []
for p in range(5, 100, 5):
    validi.append({
        "claim": f"Riduce le emissioni di CO₂ del {p}% rispetto al modello precedente",
        "categoria": "Valido",
        "spiegazione": "Quantificato e verificabile con benchmark di confronto"
    })

# 3) Genera esempi “Ingannevole” per riduzioni senza benchmark
ingannevoli = []
for p in range(5, 100, 5):
    ingannevoli.append({
        "claim": f"Riduce le emissioni di CO₂ del {p}%",
        "categoria": "Ingannevole",
        "spiegazione": "Manca riferimento comparativo"
    })

# 4) Genera esempi “Ambiguo” per packaging con percentuale ma senza certificazione
ambigui = []
for p in range(10, 100, 10):
    ambigui.append({
        "claim": f"Packaging con {p}% plastica riciclata",
        "categoria": "Ambiguo",
        "spiegazione": "Potenzialmente valido ma mancano dettagli su certificazione"
    })

# 5) Qualche esempio extra miscelato
extra = [
    {"claim":"Compatibile con energy-saving","categoria":"Irrilevante","spiegazione":"Non misura impatto ambientale"},
    {"claim":"Certificazione LEED per efficienza energetica","categoria":"Valido","spiegazione":"Standard LEED riconosciuto"},
    {"claim":"Production carbon positive al 120%","categoria":"Ingannevole","spiegazione":"Percentuale impossibile da verificare"},
    {"claim":"Ammortamento emissioni garantito","categoria":"Marketing","spiegazione":"Termine vago e promozionale"},
    {"claim":"Senza microplastiche","categoria":"Ambiguo","spiegazione":"Claim da chiarire e dimostrare"},
    {"claim":"Prodotto cruelty free certificato da Leaping Bunny","categoria":"Valido","spiegazione":"Certificazione terza parte riconosciuta"},
    {"claim":"Auto 100% ecologica secondo LCA","categoria":"Valido","spiegazione":"Basato su analisi del ciclo di vita"},
    {"claim":"Eco-friendly testato in laboratorio","categoria":"Valido","spiegazione":"Supportato da test di terza parte"},
    {"claim":"100% sostenibile","categoria":"Ingannevole","spiegazione":"Troppo generico e non misurabile"}
]

# 6) Combina e rimuovi eventuali duplicati
df_new = pd.DataFrame(validi + ingannevoli + ambigui + extra)
df_extended = pd.concat([df, df_new], ignore_index=True).drop_duplicates(subset=["claim","categoria"])

# 7) Salva il CSV esteso
df_extended.to_csv("green_claims_semantic_dataset_extended.csv", index=False, encoding="utf-8")

print(f"Dataset esteso creato: {len(df_extended)} righe totali")

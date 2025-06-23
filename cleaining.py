import pandas as pd

df = pd.read_csv("green_claims_training_dataset_doc2.csv", quotechar='"', skipinitialspace=True)
# 1) Trim whitespace
df['Claim']   = df['Claim'].str.strip()
df['Support'] = df['Support'].str.strip()
# 2) Unifica certificazioni
cert_map = {"EN13432":"EN 13432", "ISO14021":"ISO 14021"}
for old, new in cert_map.items():
    df['Support'] = df['Support'].str.replace(old, new)
# 3) Elimina duplicati
df = df.drop_duplicates()
# 4) Verifica distribuzione label
print(df['Label'].value_counts())
df.to_csv("green_claims_training_dataset_doc2_cleaned.csv", index=False, quotechar='"')
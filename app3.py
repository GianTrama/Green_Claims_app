import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
import torch
import numpy as np
import joblib
import pandas as pd
import re

# === CONFIG STREAMLIT ===
st.set_page_config(page_title="Green Claim Checker", layout="centered")

# === BERT ===
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
    model = AutoModel.from_pretrained("dbmdz/bert-base-italian-uncased")
    return tokenizer, model

tokenizer, model = load_bert()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# === CARICA IL MODELLO DOCUMENTALE PRE-ALLENATO ===
clf_doc = joblib.load("document_clf2.joblib")

def valuta_claim_documentale(claim_input, support_input): #funzione che sfrutta ML
    vec = np.concatenate([get_embedding(claim_input), get_embedding(support_input)])
    print(f"Sto valutando il claim tramite il modello ML \n Claim_input:{claim_input}\n") #Stampa di debug
    print(f"Supporto documentale: {support_input}\n Vec:{vec}")
    pred = clf_doc.predict([vec])[0] #analisi della documentazione di supporto
    proba = clf_doc.predict_proba([vec])[0][pred]
    if pred == 0:
        return ("✅ Conforme alla Direttiva UE",
                "Claim conforme a standard noti e documentato.",
                proba)
    else:
        return ("🟠 Rischio di greenwashing",
                "Claim potenzialmente vago, non verificabile o fuorviante.",
                proba)

# === ANALISI SEMANTICA AVANZATA ===
semantic_clf_5class = joblib.load("semantic_clf_5class2.joblib")
reverse_map = joblib.load("label_map2.joblib")

def valuta_chiarezza_avanzata(claim_text):
    emb = get_embedding(claim_text)
    pred = semantic_clf_5class.predict([emb])[0]
    categoria = reverse_map[pred]
    spiegazioni = {
        "Valido": "✅ Claim chiaro, quantificato e verificabile.",
        "Ambiguo": "🟡 Claim vago.",
        "Ingannevole": "🔴 Claim fuorviante o non realistico.",
        "Irrilevante": "⚪ Claim che non riguarda l'ambiente.",
        "Marketing": "📢 Claim promozionale o non tecnico."
    }
    return categoria, spiegazioni[categoria]

#-------------------------------------------------------------------------------

# === FUNZIONE CHE GENERA CLAIM E DOCUMENTO DA FORM ===
def genera_claim_e_doc(
    affermazione, parte_prodotto, percentuale, certificazioni,
    esistenza_report, riguarda_carbon_neutral, base_neutralita,
    ha_piano_riduzione, verifica_indipendente, report_pubblico,
    uso_logo_verde, logo_certificato
):
    # === Costruzione del testo del claim ===
    claim_parts = []
    # if riguarda_carbon_neutral == "Sì":
    #     claim_parts.append(affermazione + ":"+ "Prodotto carbon neutral")
    # else:
    claim_parts.append(affermazione +":")

    if percentuale:
        claim_parts.append(f"({percentuale})")
    if parte_prodotto != "Tutto il prodotto":
        claim_parts.append(f"relativo a {parte_prodotto.lower()}")

    if certificazioni:
        claim_parts.append("certificato " + ", ".join(certificazioni))

    claim_test = " ".join(claim_parts)

    # === Costruzione della documentazione di supporto ===
    doc_parts = []
    if certificazioni:
        doc_parts.append("Certificato " + ", ".join(certificazioni)+".")

    if esistenza_report == "Sì":
        doc_parts.append("È disponibile un report ufficiale.")
    if riguarda_carbon_neutral == "Sì":
        if base_neutralita:
            doc_parts.append(f"Basato su {base_neutralita.lower()}.")
        if ha_piano_riduzione == "No":
            doc_parts.append("Nessun piano di riduzione.")
        if verifica_indipendente == "No":
            doc_parts.append("Nessuna verifica indipendente.")
    if report_pubblico == "No":
        doc_parts.append("Il report non è pubblico.")
    if uso_logo_verde == "Sì":
        if logo_certificato == "No":
            doc_parts.append("Logo verde non certificato.")
        else:
            doc_parts.append("Logo ambientale certificato presente.")

    doc_test = " ".join(doc_parts)
    return claim_test, doc_test #La funzione ritorna il "Claim inserito","Documentazione di supporto"

#N.B. Il controllo semantico va fatto esclusivamente sul claim dichiarato dall'utente e non sul "Claim inserito" che è stato modificato dal programma.

# === STREAMLIT UI ===
st.title("🌿 Verifica Green Claim - Tool AI")
st.markdown("Rispondi alle domande per valutare la validità e la chiarezza del tuo green claim.")

with st.form("claim_form"):
    affermazione = st.text_input("1. Inserisci il tuo claim ambientale (es. '100% riciclabile')")
    parte_prodotto = st.selectbox("2. Quale parte del prodotto riguarda?", [
        "Tutto il prodotto", "Solo l'imballaggio", "Altra parte"
    ])
    percentuale = st.text_input("3. Specifica una percentuale se applicabile (es. 80%)")
    certificazioni = st.multiselect("4. Quali certificazioni hai?", [
        "ISO 14001", "ISO 14024", "ISO 14040", "ISO 14064", "ISO 14021",
        "FSC", "Ecolabel", "EMAS", "PAS 2060", "EN 13432", "ASTM D6400","GHG Protocol"
    ])
    esistenza_report = st.radio("5. Esiste un report a supporto?", ["Sì", "No"])
    riguarda_carbon_neutral = st.radio("6. Il claim riguarda la neutralità climatica?", ["Sì", "No"])

    base_neutralita = ha_piano_riduzione = None
    if riguarda_carbon_neutral == "Sì":
        base_neutralita = st.selectbox("7. Base della neutralità?", [
            "Riduzioni dirette", "Compensazioni", "Entrambi"
        ])
        ha_piano_riduzione = st.radio("8. Hai un piano di riduzione verificato?", ["Sì", "No"])

    verifica_indipendente = st.radio("9. Verifica da ente indipendente?", ["Sì", "No"])
    report_pubblico = st.radio("10. Il report è pubblico?", ["Sì", "No"])
    uso_logo_verde = st.radio("11. Usi un logo/marchio ambientale?", ["Sì", "No"])
    logo_certificato = None
    if uso_logo_verde == "Sì":
        logo_certificato = st.radio("12. Il logo è certificato da un ente riconosciuto?", ["Sì", "No"])

    uploaded_file = st.file_uploader("13. (Facoltativo) Carica un file PDF con prove scientifiche")

    submitted = st.form_submit_button("🔎 Analizza Claim")

if submitted:
    # 1) Genera claim_test e doc_test
    claim_test, doc_test = genera_claim_e_doc(
        affermazione, parte_prodotto, percentuale, certificazioni,
        esistenza_report, riguarda_carbon_neutral, base_neutralita,
        ha_piano_riduzione, verifica_indipendente, report_pubblico,
        uso_logo_verde, logo_certificato
    )

    # 2) Mostra il claim inserito
    st.subheader("📌 Claim inserito")
    st.write(affermazione)
    # st.write(claim_test)

    # 3) Mostra la documentazione generata
    st.subheader("📄 Documentazione di supporto")
    st.write(doc_test)

    # === CONTROLLO DOCUMENTALE (PRIMO LIVELLO) ===

    # 3.a) Logo sì ma non certificato → NON conforme
    cond_logo_scelto = isinstance(uso_logo_verde, str) and uso_logo_verde.strip().lower().startswith("s")
    cond_logo_non_cert = isinstance(logo_certificato, str) and logo_certificato.strip().lower().startswith("n")

    if cond_logo_scelto and cond_logo_non_cert:
        esito_doc = "🟠 Rischio di greenwashing"
        motivo_doc = "🔴 Hai dichiarato di usare un logo/marchio ambientale, ma NON è certificato da un ente riconosciuto."
        conf_doc = 1.00

    else:
        # 3.b) Controlli “riciclabile / carbon neutral / biodegradabile” senza certificazione specifica
        claim_lower = claim_test.lower()
        doc_lower   = doc_test.lower()

        errore_cert = False

        # Se il claim parla di “riciclabile” → serve ISO 14021
        if "riciclabile" in claim_lower:
            if not ("iso 14021" in doc_lower):
                errore_cert = True
                motivo_doc = "🔴 Claim ‘riciclabile’ senza certificazione ISO 14021 nel supporto."

        # Se il claim parla di “carbon neutral” → serve ISO 14064 o PAS 2060
        if not errore_cert and "carbon neutral" in claim_lower:
            if not ("iso 14064" in doc_lower or "pas 2060" in doc_lower):
                errore_cert = True
                motivo_doc = "🔴 Claim ‘carbon neutral’ senza certificazione ISO 14064 o PAS 2060 nel supporto."

        # Se il claim parla di “biodegradabile” → serve EN 13432
        if not errore_cert and "biodegradabile" in claim_lower:
            if not "en 13432" in doc_lower:
                errore_cert = True
                motivo_doc = "🔴 Claim ‘biodegradabile’ senza certificazione EN 13432 nel supporto."

        if errore_cert:
            esito_doc = "🟠 Rischio di greenwashing"
            conf_doc = 1.00
        else:
            # 3.c) Passa al modello documentale vero e proprio (BERT+RF)
            esito_doc, motivo_doc, conf_doc = valuta_claim_documentale(affermazione, doc_test) #ho modificato claim_test con affermazione.

            

    # 4) Visualizza il risultato del controllo documentale
    st.subheader("📊 Valutazione del rischio documentale")
    # Debug temporaneo per controllare esito_doc esatto:
    # st.write("DEBUG → esito_doc:", esito_doc)
    if "Rischio" in esito_doc:
        st.error(f"{esito_doc}\n\n📌 Motivazione: {motivo_doc}")
    else:
        st.success(f"{esito_doc}\n\n📌 Motivazione: {motivo_doc}")
    st.caption(f"🔍 Confidenza documentale: {conf_doc:.2f}")

    # 5) SOLO SE “Conforme” AL DOCUMENTO → PASSA ALL’ANALISI SEMANTICA
    
    if "Conforme" in esito_doc:
        categoria_sem, motivazione_sem = valuta_chiarezza_avanzata(affermazione) #modificato con solo il claim scritto dall'azienda
        print(f"claim: {affermazione} \n Analisi semantica: {categoria_sem}\n motivazione: {motivazione_sem}")
        # === ⛔ CONTROLLO AVANZATO: Parole “nonsense” ===

        parole_nonsense = [
            "magico", "volante", "incantato", "miracolo",
            "eco love", "super green", "mistico", "futuro perfetto"
        ]
        if any(term in claim_test.lower() for term in parole_nonsense):
            categoria_sem = "Ingannevole"
            motivazione_sem = "🔴 Claim contiene parole prive di senso ambientale ('nonsense'), classificato come ingannevole."
            #Blocco del flusso
            st.subheader("🔍 Analisi semantica avanzata")
            st.info(f"**{categoria_sem}** — {motivazione_sem}")
            st.stop()

        #BLOCCO DI CONTROLLO SEMPLICE

        # === 📌 CONTROLLO AVANZATO: Percentuali incoerenti ===
        def extract_percentuali(text):
            return [int(x) for x in re.findall(r"(\d{1,3})\s*%", text)]

        percentuali_claim = extract_percentuali(claim_test)
        try:
            percentuale_valore = int(re.sub(r"[^\d]", "", percentuale)) if percentuale else None
        except:
            percentuale_valore = None

        if percentuali_claim and percentuale_valore is not None:
            for pc in percentuali_claim:
                if abs(pc - percentuale_valore) > 5:
                    categoria_sem = "Ingannevole"
                    motivazione_sem = "🔴 Percentuale dichiarata nel claim incoerente con quella indicata nel form."

        # === 📌 CONTROLLO: claim con “riduzione” senza riferimento comparativo ===
        if any(kw in affermazione.lower() for kw in ["riduzione", "ridotte", "abbattimento"]):
            if not any(term in affermazione.lower() for term in ["rispetto a", "baseline", "modello", "anno", "comparato"]):
                categoria_sem = "Ingannevole"
                motivazione_sem = "🔴 Claim sulla riduzione privo di riferimento comparativo, classificato come ingannevole."

        # === 📄 CONTROLLO: parole “eco”, “naturale” senza prove caricate ===
        if any(term in affermazione.lower() for term in ["eco", "ecologico", "naturale"]):
            print(affermazione.lower())
            if uploaded_file is None:
                categoria_sem = "Ingannevole"
                motivazione_sem = "🔴 Claim con riferimenti a 'ecologico' o 'naturale' privo di prove scientifiche caricate."

        # === 🔄 CONTROLLO: se base_neutralita = "Riduzioni dirette" ma ha_piano_riduzione = "No" ===
        if (riguarda_carbon_neutral == "Sì" and
            base_neutralita == "Riduzioni dirette" and
            ha_piano_riduzione == "No"):
            categoria_sem = "Ambiguo"
            motivazione_sem = "🟡 Hai dichiarato un piano di riduzioni dirette, ma non hai un piano verificato: claim ambiguo."

        # === 📜 CONTROLLO: coerenza claim–certificazioni ===
    #-----------------------------------------------------------------------------
        # certificazioni_richieste = {
        #     "ecologico": ["Ecolabel", "ISO 14024", "EMAS"],
        #     "carbon": ["ISO 14064", "PAS 2060"],
        #     "riciclabile": ["FSC", "ISO 14021"],
        #     "biodegradabile": ["EN 13432", "ASTM D6400"]
        # }
        # for keyword, richieste in certificazioni_richieste.items(): 
        #     if keyword in claim_test.lower():
        #         if not any(richiesta.lower() in [cert.lower() for cert in certificazioni] for richiesta in richieste):
        #             categoria_sem = "Ingannevole"
        #             motivazione_sem = f"🔴 Claim '{keyword}' senza certificazioni coerenti: attese {', '.join(richieste)}."

#-----------------------------------------------------------------------------------------

        # === Output analisi semantica avanzata ===
        st.subheader("🔍 Analisi semantica del claim")
        st.info(f"**{categoria_sem}** — {motivazione_sem}")

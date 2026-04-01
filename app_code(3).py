
import streamlit as st
import numpy as np
import os
import joblib
import pandas as pd
import plotly.graph_objects as go

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from xgboost import XGBClassifier, XGBRegressor

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="AI Drug Discovery", layout="wide")
st.title("🧪 AI Drug Discovery Platform")

BASE_DIR = os.path.dirname(__file__)

# ===============================
# FEATURE FUNCTION
# ===============================
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp = np.array(fp)

    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
    ]

    while len(desc) < 36:
        desc.append(0)

    return np.concatenate([fp, desc])

# ===============================
# LOADERS
# ===============================
def load_xgb_json(model_class, filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.warning(f"Missing: {filename}")
        return None
    model = model_class()
    model.load_model(path)
    return model

def load_pkl(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.warning(f"Missing: {filename}")
        return None
    return joblib.load(path)

# ===============================
# LOAD MODELS
# ===============================
targets = ["SERT","DAT","D2","D3","D4","5HT1A","5HT6","5HT7"]

ic50_models = {}
for t in targets:
    clf = load_xgb_json(XGBClassifier, f"{t}_clf.json")
    reg = load_xgb_json(XGBRegressor, f"{t}_reg.json")
    if clf and reg:
        ic50_models[t] = (clf, reg)

tox_model = load_xgb_json(XGBClassifier, "tox_model.json")
compat_model = load_pkl("compat_model.pkl")

# ===============================
# PREDICTIONS
# ===============================
def predict_ic50(smiles):
    features = smiles_to_features(smiles)
    if features is None:
        return None

    features = features.reshape(1, -1)
    results = {}

    for name, (clf, reg) in ic50_models.items():
        prob = clf.predict_proba(features)[0][1]

        if prob > 0.5:
            pic50 = reg.predict(features)[0]
            ic50 = 10 ** (-pic50) * 1e9

            results[name] = (pic50, ic50, prob)
        else:
            results[name] = (None, None, prob)

    return results

def predict_toxicity(smiles):
    if tox_model is None:
        return "Model Missing"

    features = smiles_to_features(smiles)
    if features is None:
        return "Invalid"

    pred = tox_model.predict(features.reshape(1,-1))[0]
    return "Toxic" if pred == 1 else "Safe"

def predict_compatibility(drug, exc):
    if compat_model is None:
        return "Missing", 0

    d = smiles_to_features(drug)
    e = smiles_to_features(exc)

    if d is None or e is None:
        return "Invalid", 0

    combined = np.concatenate([d, e]).reshape(1, -1)

    prob = compat_model.predict_proba(combined)[0][1]
    return ("Compatible" if prob > 0.5 else "Incompatible"), prob

# ===============================
# EXCIPIENTS
# ===============================
excipients = {
    "Lactose": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O",
    "Ethanol": "CCO",
    "Glycerol": "C(C(CO)O)O",
    "Propylene glycol": "CC(O)CO",
    "Sodium benzoate": "C1=CC=C(C=C1)C(=O)[O-].[Na+]",
    "Sucrose": "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](CO)O[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O",
    "Mannitol": "C(C(C(C(C(CO)O)O)O)O)O"
}

# ===============================
# INPUT UI
# ===============================
col1, col2 = st.columns(2)

with col1:
    smiles = st.text_input("🧬 Enter Drug SMILES")

with col2:
    excipient_name = st.selectbox("💊 Select Excipient", list(excipients.keys()))

run_tox = st.checkbox("⚠️ Predict Toxicity")
run_compat = st.checkbox("⚗️ Predict Compatibility")

# ===============================
# RUN BUTTON
# ===============================
if st.button("🚀 Run Prediction"):

    if not smiles:
        st.error("Enter SMILES")
    else:
        st.subheader("📊 Results")

        # IC50
        results = predict_ic50(smiles)

        active_targets = []
        pic50_values = []

        st.write("### 🎯 Target Activity")

        cols = st.columns(4)

        for i, (t, (pic50, ic50, prob)) in enumerate(results.items()):
            with cols[i % 4]:
                if pic50:
                    st.success(f"{t}
pIC50={round(pic50,2)}
IC50={round(ic50,1)} nM")
                    active_targets.append(t)
                    pic50_values.append(pic50)
                else:
                    st.info(f"{t}
Inactive
Conf={round(prob,2)}")

        # Radar Plot
        if active_targets:
            st.write("### 📈 Multi-target Profile")

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=pic50_values,
                theta=active_targets,
                fill='toself'
            ))

            fig.update_layout(polar=dict(radialaxis=dict(visible=True)))

            st.plotly_chart(fig, use_container_width=True)

        # Toxicity
        if run_tox:
            st.write("### ⚠️ Toxicity")
            st.success(predict_toxicity(smiles))

        # Compatibility
        if run_compat:
            st.write("### ⚗️ Compatibility")
            label, prob = predict_compatibility(smiles, excipients[excipient_name])
            st.success(f"{label} (Confidence: {round(prob,2)})")

    
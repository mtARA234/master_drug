
import streamlit as st
import numpy as np
import os
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from xgboost import XGBClassifier

# ===============================

# FEATURE SIZE (FIXED)

# ===============================

FEATURE_SIZE = 2084

# ===============================

# FEATURE GENERATION

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

# SAFE LOADERS

# ===============================

def load_pkl(name):
    if not os.path.exists(name):
        return None
    return joblib.load(name)

def load_xgb_json(name):
    if not os.path.exists(name):
        return None
    model = XGBClassifier()
    model.load_model(name)
    return model

# ===============================

# TARGETS

# ===============================

targets = ["SERT","DAT","D2","D3","D4","5HT1A","5HT6","5HT7"]

# ===============================

# EXCIPIENTS

# ===============================

excipients = {
    "Ethanol": "CCO",
    "Glycerol": "C(C(CO)O)O",
    "Propylene glycol": "CC(O)CO",
    "Mannitol": "C(C(C(C(C(CO)O)O)O)O)O",
    "Sucrose": "C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O",
    "Citric acid": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
    "Sodium benzoate": "C1=CC=C(C=C1)C(=O)[O-].[Na+]",
    "Polysorbate": "CC(C)CC(C(=O)OCC(CO)O)O",
    "Starch": "C(C1C(C(C(C(O1)O)O)O)O)O"
}

# ===============================

# UI

# ===============================

st.title("🧪 AI Drug Discovery Platform")

smiles = st.text_input("Enter Drug SMILES")

target_choice = st.selectbox("Select Target", targets)

use_tox = st.checkbox("Predict Toxicity")
use_compat = st.checkbox("Predict Compatibility")

excipient_choice = None
if use_compat:
    excipient_choice = st.selectbox("Select Excipient", list(excipients.keys()))

# ===============================

# RUN

# ===============================

if st.button("Run Prediction"):


    features = smiles_to_features(smiles)

    if features is None:
        st.error("Invalid SMILES")
        st.stop()

    features = features.reshape(1, -1)

    # ===============================
    # IC50
    # ===============================
    st.subheader("📊 IC50 Prediction")

    reg_model = load_pkl(f"{target_choice}_reg.pkl")

    if reg_model:
        pic50 = reg_model.predict(features)[0]
        ic50 = 10 ** (-pic50) * 1e9

        st.success(f"{target_choice} → pIC50 = {pic50:.2f}")
        st.write(f"IC50 = {ic50:.2f} nM")
    else:
        st.warning(f"{target_choice}_reg.pkl not found")

    # ===============================
    # TOXICITY (OPTIONAL)
    # ===============================
    if use_tox:
        st.subheader("☠️ Toxicity")

        tox_model = load_pkl("tox_model.pkl")

        if tox_model:
            pred = tox_model.predict(features)[0]
            st.success("Toxic" if pred == 1 else "Non-toxic")
        else:
            st.warning("tox_model.pkl missing")

    # ===============================
    # COMPATIBILITY (OPTIONAL)
    # ===============================
    if use_compat:
        st.subheader("⚖️ Compatibility")

        compat_model = load_xgb_json("compatibility_xgb.json")

        exc_smiles = excipients[excipient_choice]
        exc_feat = smiles_to_features(exc_smiles)

        if compat_model and exc_feat is not None:

            combined = np.concatenate([features[0], exc_feat]).reshape(1, -1)

            pred = compat_model.predict(combined)[0]

            st.success("Compatible" if pred == 1 else "Incompatible")

        else:
            st.warning("compatibility_xgb.json missing or invalid")
      
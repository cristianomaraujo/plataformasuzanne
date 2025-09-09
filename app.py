# app.py — Predição de Disfagia (Random Forest / scikit-learn)
import os, glob
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ----------------------------
# Configuração básica
# ----------------------------
st.set_page_config(page_title="Predição de Disfagia", layout="centered")
st.markdown("<style>.block-container{padding-top:0.5rem;}</style>", unsafe_allow_html=True)

HEADER_IMAGE = "figura.jpg"  # opcional
if os.path.exists(HEADER_IMAGE):
    st.image(HEADER_IMAGE, use_container_width=True)

st.caption("Uso acadêmico. As predições não substituem o julgamento clínico.")

# Classe positiva (presença do desfecho neste dataset)
POSITIVE_CLASS = 2  # 1 = sem disfagia, 2 = com disfagia

# ----------------------------
# Localizar e carregar modelo
# ----------------------------
def listar_modelos():
    preferido = "bestmodel_randomforest.pkl"
    itens = []
    if os.path.exists(preferido):
        itens.append(preferido)
    itens += [p for p in sorted(glob.glob("*.pkl")) if p != preferido]
    return itens

@st.cache_resource(show_spinner=False)
def carregar_modelo(path: str):
    if not os.path.exists(path):
        st.error(f"Arquivo de modelo não encontrado: {path}")
        raise FileNotFoundError(path)
    try:
        m = joblib.load(path)  # requer scikit-learn instalado
        return m
    except ModuleNotFoundError as e:
        st.error(
            "Este arquivo .pkl requer pacotes não instalados neste ambiente.\n\n"
            f"Detalhe: {e}\n\n"
            "Para modelos RandomForest do scikit-learn, instale:\n"
            "`pip install scikit-learn joblib`"
        )
        raise
    except Exception as e:
        st.exception(e)
        raise

model_files = listar_modelos()
if not model_files:
    st.error("Nenhum arquivo de modelo encontrado (procuro *.pkl na pasta).")
    st.stop()

modelo_escolhido = st.sidebar.selectbox("Modelo", model_files, index=0)
model = carregar_modelo(modelo_escolhido)
st.sidebar.caption(f"Carregado: {os.path.basename(modelo_escolhido)}")

# ----------------------------
# Nomes das features esperadas
# ----------------------------
feature_names = None
for attr in ("feature_names_in_", "feature_names_", "features_"):
    if hasattr(model, attr):
        try:
            vals = list(getattr(model, attr))
            if vals:
                feature_names = [str(v) for v in vals]
                break
        except Exception:
            pass

# Fallback alinhado à sua base — ajuste se seu RF não usar "sexo"
if not feature_names:
    feature_names = ["sexo", "idade", "glasgow", "eat10", "fois"]

# Remover qualquer alvo por engano
TARGET_CANDIDATES = {"disfagia", "target", "alvo", "label", "classe", "outcome", "y"}
alvos_no_modelo = [f for f in feature_names if f.lower() in TARGET_CANDIDATES]
if alvos_no_modelo:
    st.error(
        "O modelo parece incluir o desfecho nas features "
        f"({', '.join(alvos_no_modelo)}). Re-treine sem o alvo em X e salve novamente."
    )
    st.stop()

features_ui = feature_names[:]  # todas as features são de entrada neste app

# ----------------------------
# Diagnóstico rápido (classes)
# ----------------------------
classes_ = list(getattr(model, "classes_", [])) if hasattr(model, "classes_") else None
with st.sidebar.expander("Diagnóstico", expanded=False):
    st.write("Classes do modelo:", classes_)
    st.write("Classe positiva utilizada:", POSITIVE_CLASS)

# ----------------------------
# Entradas (com 'Sexo' por extenso)
# ----------------------------
SEXO_LABELS = ["Masculino", "Feminino"]  # 1=Masculino, 2=Feminino
MAP_SEXO    = {"Masculino": 1, "Feminino": 2}

st.subheader("Entradas clínicas para triagem de Disfagia usando Machine Learning")

def defaults(nome):
    n = nome.lower()
    if "idade" in n:   return 32, 0, 120
    if "glasgow" in n: return 15, 3, 15
    if "eat10" in n:   return 0, 0, 40
    if "fois" in n:    return 7, 1, 8
    return 0, 0, 100

inputs = {}
cols = st.columns(2) if len(features_ui) > 1 else [st]
for i, fname in enumerate(features_ui):
    with cols[i % len(cols)]:
        if fname.lower() == "sexo":
            sexo_lbl = st.selectbox("Sexo", SEXO_LABELS, index=0, key="inp_sexo")
            inputs[fname] = MAP_SEXO[sexo_lbl]  # mapeia para 1/2 conforme o treino
        else:
            default, minv, maxv = defaults(fname)
            inputs[fname] = st.number_input(
                fname, min_value=minv, max_value=maxv, value=default, step=1, key=f"inp_{fname}"
            )

# ----------------------------
# Botão de cálculo (recalculável)
# ----------------------------
if st.button("Calcular", type="primary"):
    try:
        # Monta X na ORDEM que o modelo espera
        row = {}
        for c in feature_names:
            if c in inputs:
                row[c] = int(inputs[c])
            else:
                row[c] = 0  # segurança para colunas inesperadas (raro)
        X = pd.DataFrame([row], columns=feature_names)

        # Probabilidade da classe "presença de disfagia" (classe 2 neste dataset)
        proba = model.predict_proba(X)
        proba = np.array(proba)[0]  # (n_classes,)

        # Seleção correta do índice da classe positiva = 2
        pos_idx = len(proba) - 1  # fallback
        try:
            if classes_ and POSITIVE_CLASS in classes_:
                pos_idx = classes_.index(POSITIVE_CLASS)
        except Exception:
            pass

        p = float(proba[pos_idx])

        st.subheader("Predição")
        st.metric("Probabilidade de disfagia (classe 2)", f"{p:.0%}")
        st.write(f"Existe a probabilidade de **{p:.0%}** de presença de **disfagia**.")

    except Exception as e:
        st.error(f"Ocorreu um erro ao calcular a predição: {e}")

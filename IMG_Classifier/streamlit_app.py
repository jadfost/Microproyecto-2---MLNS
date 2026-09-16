#  We ensure proper path handling in Python
import Definitions
import pandas as pd
import streamlit as st

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Clasificador de ODS", page_icon="🌍"
)

### My vars

@st.cache_resource
def get_controller():
    return ModelController()


ctrl = get_controller()

### My UI starting here

st.title("🌍 Clasificador de textos según los ODS")
st.write(
    "Escribe o pega un texto libre (una noticia, un párrafo de un reporte, una opinión "
    "ciudadana, etc.) y el modelo predecirá con cuál de los 17 Objetivos de Desarrollo "
    "Sostenible (ODS) se relaciona más."
)

with st.form(key="my_form"):
    input_text = st.text_area(
        "Texto a clasificar",
        height=180,
        placeholder="Por ejemplo: 'El acceso a agua potable sigue siendo limitado en "
                    "las zonas rurales, lo que afecta la salud de miles de familias...'",
    )

    submit_button = st.form_submit_button(label="Predecir ODS")

if submit_button:
    if input_text is None or input_text.strip() == "":
        st.warning("⚠️ Por favor ingresa un texto antes de predecir.")
    else:
        st.session_state["input_text"] = input_text

texto_guardado = st.session_state.get("input_text")

if texto_guardado:
    st.caption("✅ Texto analizado")
    st.write(texto_guardado)

    ods_pred, ods_name, probabilidades = ctrl.predict(texto_guardado)
    top_prob = probabilidades[ods_pred]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.caption("🎯 Predicción")
        st.metric("ODS predicho", f"ODS {ods_pred}")
        st.metric("Confianza", f"{top_prob:.1%}")

    with col2:
        st.caption("🗣 Objetivo de Desarrollo Sostenible")
        st.success(f"**ODS {ods_pred} — {ods_name}**")

    st.caption("📊 Probabilidad estimada por ODS (top 5)")
    top5 = dict(list(probabilidades.items())[:5])
    df_probs = pd.DataFrame({
        "ODS": [f"ODS {k}" for k in top5.keys()],
        "Probabilidad": list(top5.values()),
    }).set_index("ODS")
    st.bar_chart(df_probs)

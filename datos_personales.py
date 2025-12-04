import streamlit as st
st.set_page_config(page_title="Datos Personales", layout="centered")

st.markdown("""
<style>
    .card {
        background: linear-gradient(180deg, #a1d6e2, #a1d6e2);
        border-radius: 12px;
        padding: 18px;
        color: #e6f6ff;
        box-shadow: 0 6px 18px rgba(0,0,0,0.5);
    }
    .label {
        color: #afd7ff;
        font-weight: 600;
    }
    .metric {
        font-size: 20px;
        font-weight: 700;
        color: #d8f3ff;
    }
    .small {
        color: #bcdff7;
        font-size: 13px;
    }
    .two-col { display:flex; gap:12px; }
    .two-col > div { flex: 1; }
</style>
""", unsafe_allow_html=True)

st.title("Datos Personales")
st.markdown("Rellena los datos personales para unirte a nuestra comunidad.")

with st.form("perfil_gym_form", clear_on_submit=False):
    st.header("Datos personales")
    col1, col2 = st.columns(2)

    with col1:
        nombre = st.text_input("Nombre completo", max_chars=80)
        fecha_nac = st.date_input("Fecha de nacimiento (obligatorio)")
        sexo = st.selectbox("Sexo", ["Masculino","Femenino"])

    with col2:
        edad = st.number_input("Edad (años)", min_value=12, max_value=80, value=16)
        altura_cm = st.number_input("Altura (cm)", min_value=100, max_value=270, value=170)
        peso_kg = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0, format="%.1f")

    st.header("Información física y objetivos")
    objetivo = st.selectbox("Objetivo principal", [
        "Gain muscle/weight",
        "lose weight"
    ])
    experiencia = st.selectbox("Experiencia en gimnasio", [
        "Beginner (0-6 meses)",
        "Intermediate (6-12 meses)",
        "Expert (>12 meses)"
    ])


    #st.header("Preferencias de entrenamiento")
    #prefentreno = st.multiselect("Tipo de entrenamiento preferido", [
        #"Fuerza",
        #"Cardio",
        #"Resistencia",
        #"Elasticidad",
        #"Yoga/Estiramientos",
        #"Mixto"
    #], default=["Fuerza"])

    enviar = st.form_submit_button("Enviar") #no hace nada de momento

st.info("Rellena el formulario para unirte a nuestra comunidad!!!")  






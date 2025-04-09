import streamlit as st
import requests

# Titre de la page
st.title("🚗 Prédiction de Consommation de Carburant")

# Texte de description
st.markdown("Entrez les caractéristiques du véhicule pour prédire la consommation en **L/100km** et **MPG**.")

# Formulaire utilisateur
weight = st.number_input("Poids (en lbs)", value=3000)
acceleration = st.number_input("Accélération", value=15.0)
displacement = st.number_input("Cylindrée (displacement)", value=200)
cylinders = st.number_input("Nombre de cylindres", value=4)
model_year = st.number_input("Année du modèle", value=76)
horsepower = st.number_input("Puissance (horsepower)", value=95)
origin = st.selectbox("Origine", options=[1, 2, 3], help="1: USA, 2: Europe, 3: Japon")

# Bouton de prédiction
if st.button("🔮 Prédire"):
    url = "http://localhost:8000/predict/"  # Assure-toi que ton API Flask tourne !

    payload = {
        "weight": weight,
        "acceleration": acceleration,
        "displacement": displacement,
        "cylinders": cylinders,
        "model_year": model_year,
        "horsepower": horsepower,
        "origin": origin
    }

    try:
        response = requests.post(url, json=payload)
        result = response.json()

        if "error" in result:
            st.error(f"Erreur : {result['error']}")
        else:
            st.success("✅ Prédiction réussie !")
            st.write(f"**Consommation en MPG**: {result['consommation_reelle_mpg']:.2f}")
            st.write(f"**Consommation en L/100km**: {result['consommation_reelle_l_100km']:.2f}")
            st.info(result["commentaire_conversion"])
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")

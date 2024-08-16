import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
# module python utilisé pour enregistrer le modèle ML
import joblib 

# Fonction de prétraitement des données
def treatInput(test):
    try:
        categorical_columns = ['Region', 'Brand','Power',"Gas"]

        toFit = pd.read_csv("variables.csv")

        # Initialisation du OneHotEncoder
        encoder = OneHotEncoder(sparse=False)
        # Application de l'encodage

        toFitdata = encoder.fit_transform(toFit[categorical_columns])
        encoded_data = encoder.transform(test[categorical_columns])

        # Création d'un DataFrame avec les nouvelles colonnes encodées
        toFitdf = pd.DataFrame(toFitdata, columns=encoder.get_feature_names(categorical_columns))
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(categorical_columns))

        # Combinaison avec les autres colonnes non catégorielles
        toFit = pd.concat([toFit.drop(columns=categorical_columns), toFitdf], axis=1)
        X_enc = pd.concat([test.drop(columns=categorical_columns), encoded_df], axis=1)

        print(X_enc.shape)

        # Créer l'objet StandardScaler
        scaler = MinMaxScaler()
        # Ajuster le scaler aux données d'entraînement
        scaler.fit(toFit)
        X_scaled = scaler.transform(X_enc)

        X_scaled_df = pd.DataFrame(X_scaled, columns=X_enc.columns)
        # Convertir le résultat en DataFrame pour faciliter l'utilisation ultérieure
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_enc.columns)

        X_scaled_df.head()
        return X_scaled_df
    except:
        exit() 
        
    
   

# Fonction qui gère la prédiction de la prime
def predictFrequence(data):
    print(data.columns)
    try:
        # Chargement du modèle pré-entraîné C:\Users\EEIA\Desktop\Assurance_project\tweedieModel.pkl
        with open('modelfrequence.pkl', 'rb') as file:
            print(file)
            model = joblib.load(file)
            st.write("model de fréquence chargé")
            print("model de fréquence chargé",model)
            prediction = model.predict(data)
            print("prediction frequency",prediction)
            if prediction:
                return prediction[0]
    except:
        exit()


# Fonction qui gère la prédiction de la prime
def predictSeverity(data):
    print(data)
    try:
        # Chargement du modèle pré-entraîné C:\Users\EEIA\Desktop\Assurance_project\tweedieModel.pkl
        with open('modelseverite.pkl', 'rb') as file:
            print(file)
            model = joblib.load(file)
            st.write("model de sévérité chargé")
            print("model de sévérité chargé",model)
            prediction = model.predict(data)
            print("prediction severity",prediction)
            if prediction:
                return prediction[0]*19153.113869661753
    except:
        exit()

def predictPrime(data):
    frequency = predictFrequence(data)
    severity = predictSeverity(data)
    return frequency*severity

# Configuration de la page
st.set_page_config(page_title="Caractéristiques de l'Assuré et de l'Automobile", page_icon=":car:")

st.title("Formulaire de Caractéristiques de l'Assuré et de l'Automobile")

# Section 1 : Caractéristiques de l'Assuré
st.header("Caractéristiques de l'Assuré")

age = st.number_input("Âge de l'assuré", min_value=18, max_value=100, step=1, help="Entrez l'âge de l'assuré.")
identifiant_contrat = st.text_input("Identifiant du contrat", help="Entrez l'identifiant unique du contrat.")
periode_couverture = st.date_input(
    "Période de couverture de l'assurance",
    value=[date.today(), date.today().replace(year=date.today().year + 1)],
    help="Sélectionnez la période de début et de fin de la couverture."
)
region_habitation = st.selectbox(
    "Région d'habitation",
    ['Aquitaine', 'Nord-Pas-de-Calais', 'Pays-de-la-Loire',
       'Ile-de-France', 'Centre', 'Poitou-Charentes', 'Bretagne',
       'Basse-Normandie', 'Limousin', 'Haute-Normandie'],
    help="Sélectionnez la région d'habitation de l'assuré."
)
densite_population = st.slider(
    "Densité de la région (habitants par km²)",
    min_value=10, max_value=10000, step=10,
    help="Entrez la densité de la population dans la région d'habitation de l'assuré."
)

# Section 2 : Caractéristiques de l'Automobile
st.header("Caractéristiques de l'Automobile")

marque = st.selectbox(
    "Marque de l'automobile",
    ['Japanese (except Nissan) or Korean', 'Fiat', 'Opel, General Motors or Ford', 'Mercedes, Chrysler or BMW',
'Renault, Nissan or Citroen', 'Volkswagen, Audi, Skoda or Seat','other'],
    help="Sélectionnez la marque de l'automobile."
)
# puissance = st.number_input(
#     "Puissance de l'automobile (en chevaux)",
#     min_value=50, max_value=1000, step=10,
#     help="Entrez la puissance de l'automobile en chevaux."
# )
puissance = st.selectbox(
    "Puissance de l'automobile",
    ['d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
    help="Entrez la puissance de l'automobile."
)
age_vehicule = st.number_input(
    "Âge de l'automobile (en années)",
    min_value=0, max_value=50, step=1,
    help="Entrez l'âge du véhicule en années."
)
carburation = st.radio(
    "Type de carburation",
    ["Regular", "Diesel"],
    help="Sélectionnez le type de carburation de l'automobile. Regular(Essence, Electrique, Hybride)"
)

# Bouton de soumission
if st.button("Soumettre les informations"):

    st.success("Les informations ont été soumises avec succès!")
    
    # Enregistrer les données sous forme de dictionnaire
    input_data = {
        'Exposure': [((periode_couverture[1] - periode_couverture[0]).days)/365.0],  # Période en jours / 365.0
        'Power': [puissance],
        'CarAge': [age_vehicule],
        'DriverAge': [age],
        'Brand': [marque],
        'Gas': [carburation],
        'Region': [region_habitation],
        'Density': [densite_population],
    }

    input_df = pd.DataFrame(input_data)
    print(input_df)
    # Prétraiter les données avant de les passer au modèle
    input_treated = treatInput(input_df)

    print("données encodées", input_treated)

    # Calcul de la prime
    prime = predictPrime(input_treated)

    # Exemple d'affichage des données saisies :
    st.write("## Résumé des informations saisies :")
    st.write(f"**Âge de l'assuré :** {age} ans")
    st.write(f"**Identifiant du contrat :** {identifiant_contrat}")
    st.write(f"**Période de couverture :** du {periode_couverture[0]} au {periode_couverture[1]}")
    st.write(f"**Région d'habitation :** {region_habitation}")
    st.write(f"**Densité de population :** {densite_population} habitants/km²")
    st.write(f"**Marque de l'automobile :** {marque}")
    st.write(f"**Puissance de l'automobile :** {puissance} chevaux")
    st.write(f"**Âge de l'automobile :** {age_vehicule} ans")
    st.write(f"**Type de carburation :** {carburation}")

    # Affichage du résultat de la prédiction
    st.write("### Estimation de la Prime d'Assurance :")
    st.write(f"La prime d'assurance estimée pour cet assuré est de : **{prime:.2f} €**")

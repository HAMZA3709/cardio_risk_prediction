import streamlit as st
import pandas as pd
import mysql.connector
import bcrypt
from itsdangerous import URLSafeTimedSerializer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIGURATION
#pour run :  streamlit run step7_app.py

# =========================================================
st.set_page_config(page_title="Prédiction Cardio", layout="centered")

SECRET_KEY = "SECRET_KEY_123"
serializer = URLSafeTimedSerializer(SECRET_KEY)

# =========================================================
# BASE DE DONNÉES
# =========================================================
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="cardio_app"
    )

# =========================================================
# AUTHENTIFICATION
# =========================================================
def register_user(email, password):
    db = get_db()
    cursor = db.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    cursor.execute(
        "INSERT INTO users(email, password) VALUES (%s, %s)",
        (email, hashed)
    )
    db.commit()

def login_user(email, password):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()
    if user and bcrypt.checkpw(password.encode(), user["password"].encode()):
        return user
    return None

def generate_token(email):
    return serializer.dumps(email)

# =========================================================
# SAUVEGARDE DES PRÉDICTIONS
# =========================================================
def save_prediction(user_id, data, result, probability):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (user_id, age, gender, bmi, ap_hi, ap_lo,
         cholesterol, gluc, smoke, alco, active,
         result, probability)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        int(user_id),
        int(data["age"]),
        int(data["gender"]),
        float(data["bmi"]),
        int(data["ap_hi"]),
        int(data["ap_lo"]),
        int(data["cholesterol"]),
        int(data["gluc"]),
        int(data["smoke"]),
        int(data["alco"]),
        int(data["active"]),
        int(result),
        float(probability)
    ))

    db.commit()


# =========================================================
# MODÈLE ML
# =========================================================
df = pd.read_csv("data/cardio_clean.csv")

X = df.drop(columns=["cardio"])
y = df["cardio"]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================================================
# SESSION
# =========================================================
if "user" not in st.session_state:
    st.session_state.user = None

# =========================================================
# INTERFACE AUTH
# =========================================================
st.sidebar.title("🔐 Authentification")
choice = st.sidebar.selectbox("Choisir", ["Connexion", "Inscription"])

if choice == "Inscription":
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Mot de passe", type="password")

    if st.sidebar.button("Créer un compte"):
        register_user(email, password)
        token = generate_token(email)
        st.sidebar.success("Compte créé avec succès")
        st.sidebar.info(f"Lien de vérification (simulation) : {token}")

if choice == "Connexion":
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Mot de passe", type="password")

    if st.sidebar.button("Se connecter"):
        user = login_user(email, password)
        if user:
            st.session_state.user = user
            st.sidebar.success("Connexion réussie")
        else:
            st.sidebar.error("Email ou mot de passe incorrect")

# =========================================================
# APPLICATION PRINCIPALE
# =========================================================
st.title("🫀 Prédiction du risque cardiovasculaire")

if st.session_state.user:

    st.write("Veuillez saisir les informations médicales suivantes :")

    st.write("Veuillez saisir les informations médicales suivantes :")

    age = st.number_input("Âge (années)", 18, 100, 50)

    gender_label = st.selectbox("Genre", ["Femme", "Homme"])
    gender = 1 if gender_label == "Femme" else 2

    height = st.number_input("Taille (cm)", 140, 220, 170)
    weight = st.number_input("Poids (kg)", 40, 200, 70)

    ap_hi = st.number_input("Pression systolique", 80, 250, 120)
    ap_lo = st.number_input("Pression diastolique", 40, 200, 80)

    chol_label = st.selectbox("Cholestérol", ["Normal", "Au-dessus de la normale", "Élevé"])
    cholesterol = {"Normal": 1, "Au-dessus de la normale": 2, "Élevé": 3}[chol_label]

    gluc_label = st.selectbox("Glucose", ["Normal", "Au-dessus de la normale", "Élevé"])
    gluc = {"Normal": 1, "Au-dessus de la normale": 2, "Élevé": 3}[gluc_label]

    smoke_label = st.selectbox("Fumeur", ["Non", "Oui"])
    smoke = 1 if smoke_label == "Oui" else 0

    alco_label = st.selectbox("Consommation d’alcool", ["Non", "Oui"])
    alco = 1 if alco_label == "Oui" else 0

    active_label = st.selectbox("Activité physique", ["Non", "Oui"])
    active = 1 if active_label == "Oui" else 0

    bmi = weight / ((height / 100) ** 2)

    if st.button("🔍 Prédire le risque"):
        input_data = pd.DataFrame([[gender, height, weight, ap_hi, ap_lo,
                                    cholesterol, gluc, smoke, alco, active,
                                    age, bmi]], columns=X.columns)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        user_data = {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active
        }

        save_prediction(
            st.session_state.user["id"],
            user_data,
            prediction,
            probability
        )

        if prediction == 1:
            st.error(f"⚠️ Risque cardiovasculaire ÉLEVÉ (probabilité : {probability:.2f})")
        else:
            st.success(f"✅ Risque cardiovasculaire FAIBLE (probabilité : {probability:.2f})")

else:
    st.warning("Veuillez vous connecter ou créer un compte pour accéder à la prédiction.")

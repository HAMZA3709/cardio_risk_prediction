import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("data/cardio_clean.csv")

#Séparer X et y
X = df.drop(columns=['cardio'])
y = df['cardio']

#Séparer en données d’entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#. Normalisation des données
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Vérification
print("Train :", X_train_scaled.shape)
print("Test :", X_test_scaled.shape)

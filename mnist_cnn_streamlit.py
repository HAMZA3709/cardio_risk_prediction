# ======================================================
# CNN MNIST + Interface Streamlit (script unique)
# ======================================================
#run streamlit run mnist_cnn_streamlit.py
# ============================================
# CNN MNIST + Interface Streamlit (Upload image)
# ============================================

import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = "mnist_cnn_model.h5"

# ============================================
# 1. Chargement et préparation des données MNIST
# ============================================
def train_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalisation
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Redimensionnement pour CNN
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    # ============================================
    # 2. Construction du Modèle CNN
    # ============================================
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # ============================================
    # 3. Compilation et Entraînement
    # ============================================
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    st.write("🔄 Entraînement du modèle en cours...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)

    return model

# ============================================
# 4. Charger ou entraîner le modèle
# ============================================
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = train_model()

# ============================================
# 5. Interface graphique Streamlit
# ============================================
st.title("✍️ Reconnaissance de chiffres manuscrits (MNIST)")
st.write("Chargez une image contenant un chiffre manuscrit (0 à 9).")

uploaded_file = st.file_uploader(
    "📂 Charger une image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Préparation de l’image
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    st.image(image, caption="Image chargée", width=150)

    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    if st.button("🔍 Reconnaître le chiffre"):
        prediction = model.predict(img_array)
        chiffre = np.argmax(prediction)
        st.success(f"✅ Chiffre reconnu : {chiffre}")

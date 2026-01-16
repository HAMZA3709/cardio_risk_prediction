import pandas as pd

df = pd.read_csv("data/cardio_train.csv", sep=';')
df['age_years'] = (df['age'] / 365).astype(int)
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df = df[
    (df['height'] > 120) & (df['height'] < 220) &
    (df['weight'] > 30) & (df['weight'] < 200) &
    (df['ap_hi'] > 70) & (df['ap_hi'] < 250) &
    (df['ap_lo'] > 40) & (df['ap_lo'] < 200)
]
df = df.drop(columns=['id', 'age'])
print(df.head())
print(df.shape)
print(df.isnull().sum())

# Sauvegarder les données propres
df.to_csv("data/cardio_clean.csv", index=False)

print("✅ Données nettoyées sauvegardées dans data/cardio_clean.csv")

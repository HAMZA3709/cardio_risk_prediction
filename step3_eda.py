import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cardio_clean.csv")
#Distribution de l’âge
plt.figure()
plt.hist(df['age_years'], bins=30)
plt.title("Distribution de l'âge")
plt.xlabel("Âge")
plt.ylabel("Nombre de patients")
plt.show()

#IMC vs Risque cardiovasculaire
plt.figure()
sns.boxplot(x='cardio', y='bmi', data=df)
plt.title("IMC selon le risque cardiovasculaire")
plt.xlabel("Maladie cardiovasculaire (0 = Non, 1 = Oui)")
plt.ylabel("IMC")
plt.show()

#Pression artérielle vs Risque
plt.figure()
sns.boxplot(x='cardio', y='ap_hi', data=df)
plt.title("Pression systolique et risque cardiovasculaire")
plt.show()

#Matrice de corrélation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()

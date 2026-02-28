import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

file_path = r"c:\Users\zeyne\derin_ogrenme_odevler\odev-1\data.csv"

df = pd.read_csv(file_path)

if 'id' in df.columns:
    df = df.drop(columns=['id'])
if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])

# target degiskeni sayisala cevir (M=1, B=0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# ozellikler X ve hedef y
X = df.drop('diagnosis', axis=1) # diagnosis disindaki tum sutunlar
y = df['diagnosis'] # sadece diagnosis sutunu

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Veri Hazir ---")
print("Egitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# n_estimators=100: 100 farklı karar ağacı oluşturulacak demek
model = RandomForestClassifier(n_estimators=100, random_state=42)

# model eğitimi
model.fit(X_train, y_train)

# test verisi ile tahmin
y_pred = model.predict(X_test)

print("--- Model Egitildi ve Tahminler Yapildi ---")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Metrikler
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- ODEV SONUCLARI ---")
print(f"Accuracy (Dogruluk): {accuracy:.4f}")
print(f"Precision (Kesinlik): {precision:.4f}")
print(f"Recall (Duyarlilik): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nKarmasiklik Matrisi:")
print(confusion_matrix(y_test, y_pred))
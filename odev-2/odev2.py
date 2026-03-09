import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = MLPClassifier(
    hidden_layer_sizes=(8,), 
    activation='tanh', # XOR için uygun bir aktivasyon fonksiyonu
    solver='lbfgs', # küçük veri setlerinde başarılı bir optimize edici
    max_iter=1000, 
    random_state=42
)

# model eğitimi
print("Model egitiliyor, lutfen bekleyin...")
model.fit(X, y)

# sonuçlar
tahminler = model.predict(X)

print("\n*** EGITIM SONUCLARI ***")
print(f"Dogruluk Skoru: %{model.score(X, y) * 100}")

print("\n*** XOR TAHMINLERI ***")
for i in range(len(X)):
    print(f"Girdi: {X[i]} -> Gercek: {y[i]} | Tahmin: {tahminler[i]}")
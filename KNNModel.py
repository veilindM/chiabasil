from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib  # To save the model

# ✅ Load updated dataset
df = pd.read_csv("chia_basil_dataset.csv")

# ✅ Use all features (not just HSV)
X = df.drop(columns=["Class"]).values
y = df["Class"].values

# ✅ Split dataset into 80% training & 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train k-NN model
k = 3  # Try k=5, k=7 to compare results
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# ✅ Make predictions on test data
y_pred = knn.predict(X_test_scaled)

# ✅ Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Save model and scaler
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler retrained & saved!")

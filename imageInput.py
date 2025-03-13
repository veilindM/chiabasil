import cv2
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Function to extract HSV + Texture + Edges + Shape features
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None  # Handle invalid images
    
    image = cv2.resize(image, (200, 200))  
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # HSV Features (Color)
    h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)
    
    # Texture Features using GLCM (Grey Level Co-occurrence Matrix)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    # Edge Detection using Canny
    edges = cv2.Canny(gray_image, 100, 200)
    edge_ratio = np.count_nonzero(edges) / (200 * 200)
    
    # Shape Features (Contour Analysis)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    else:
        area, perimeter, circularity = 0, 0, 0  # If no contours found
    
    # Return feature vector
    return [h_mean, s_mean, v_mean, contrast, energy, edge_ratio, area, perimeter, circularity]

# Load dataset
df = pd.read_csv("chia_basil_dataset.csv")
X = df.drop(columns=["Class"]).values  # Remove only the "Class" column, keep all features
y = df["Class"].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train k-NN model
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # ✅ Load trained model & scaler
    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # ✅ Extract features from the new image
    features = extract_features(file_path)
    if features is None:
        print("⚠️ Invalid image. Please try again.")
        return
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)  # Scale features

    # ✅ Get nearest neighbor distances
    distances, indices = knn.kneighbors(features_scaled)
    avg_distance = np.mean(distances)

    threshold = 1.5  # Set an appropriate threshold
    if avg_distance > threshold:
        print("🔴 The uploaded image is classified as: Unknown")
    else:
        prediction = knn.predict(features_scaled)[0]
        print(f"🟢 The uploaded image is classified as: {prediction}")

# GUI for image upload
root = tk.Tk()
root.withdraw()  # Hide main window
print("Select an image for classification...")
classify_image()
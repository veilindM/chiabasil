import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops

# Function to extract HSV + Texture + Edges + Shape features
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.resize(image, (200, 200))  
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # ✅ HSV Features (Color)
    h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)
    
    # ✅ Texture Features using GLCM
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    glcm = graycomatrix(gray_image, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    # ✅ Edge Detection using Canny
    edges = cv2.Canny(gray_image, 100, 200)
    edge_ratio = np.count_nonzero(edges) / (200 * 200)
    
    # ✅ Shape Features (Contour Analysis)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    else:
        area, perimeter, circularity = 0, 0, 0  # If no contours found
    
    return [h_mean, s_mean, v_mean, contrast, energy, edge_ratio, area, perimeter, circularity]

# ✅ Collect dataset
image_folder = "."  # Change if necessary
data = []

for label in ["Chia", "Basil"]:
    folder_path = os.path.join(image_folder, label)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found!")
        continue
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            features = extract_features(img_path)
            if features:
                data.append([*features, label])

# ✅ Save dataset
df = pd.DataFrame(data, columns=[
    "Hue", "Saturation", "Value",        # HSV Features
    "Contrast", "Energy",                # Texture Features
    "Edge_Ratio", "Shape_Area", "Shape_Perimeter", "Circularity",  # Edge & Shape Features
    "Class"
])
df.to_csv("chia_basil_dataset.csv", index=False)

print("✅ Dataset updated successfully!")

import pandas as pd
import os
import numpy as np
import skimage

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print(skimage.__version__)

# data = []
# image_folder = "dataset_folder"

# for label in ["Chia", "Basil"]:
#     folder_path = os.path.join(image_folder, label)
#     for img_name in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_name)
#         h, s, v = extract_hsv_features(img_path)
#         data.append([h, s, v, label])

# df = pd.DataFrame(data, columns=["Hue", "Saturation", "Value", "Class"])
# df.to_csv("chia_basil_dataset.csv", index=False)

# # Example usage
# # image_path = "Basil/1.jpg"
# # h, s, v = extract_hsv_features(image_path)
# # print(f"Hue: {h}, Saturation: {s}, Value: {v}")

import os

folders = [
    "model",
    "templates",
    "static",
    "dataset/sample_data/Basmati",
    "dataset/sample_data/Jasmine",
    "dataset/sample_data/Arborio",
    "dataset/sample_data/Brown",
    "dataset/sample_data/Red"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✅ Folder structure created.")

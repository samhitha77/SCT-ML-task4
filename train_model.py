import os
import cv2
import joblib
import numpy as np

from tqdm import tqdm
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

DATASET = "dataset"
IMG_SIZE = 128

X = []
y = []

gesture_names = {}
label = 0

for folder in sorted(os.listdir(DATASET)):

    folder_path = os.path.join(DATASET, folder)

    if not os.path.isdir(folder_path):
        continue

    gesture_names[label] = folder

    print(f"\nLoading {folder}...")

    for file in tqdm(os.listdir(folder_path)):

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        feature = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            block_norm="L2-Hys"
        )

        X.append(feature)
        y.append(label)

    label += 1

X = np.array(X)
y = np.array(y)

print("\nDataset Loaded")
print("Samples :", len(X))
print("Classes :", len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(
    kernel="rbf",
    probability=True,
    C=10
)

print("\nTraining...")

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, pred))

print(classification_report(y_test, pred))

joblib.dump(model, "gesture_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(gesture_names, "gesture_names.pkl")

print("\nModel Saved!")
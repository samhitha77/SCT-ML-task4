# ✋ Rock Paper Scissors Gesture Recognition using Machine Learning

A real-time **Rock, Paper, Scissors hand gesture recognition system** built using **Python**, **OpenCV**, **HOG (Histogram of Oriented Gradients)**, and a **Support Vector Machine (SVM)** classifier.

The application detects hand gestures from a webcam feed and classifies them into one of the following classes:

- ✊ Rock
- ✋ Paper
- ✌️ Scissors

---

## 📌 Features

- Real-time webcam prediction
- Hand Region of Interest (ROI)
- Skin color segmentation using HSV
- HOG feature extraction
- SVM classifier for gesture recognition
- Live confidence score
- Automatic "No Hand Detected" when no hand is present
- Lightweight and fast inference

---

## 📂 Project Structure

```
SCT_ML_4/
│
├── dataset/
│   ├── rock/
│   ├── paper/
│   └── scissors/
│
├── train_model.py
├── predict_webcam.py
├── gesture_model.pkl
├── scaler.pkl
├── gesture_names.pkl
├── README.md
```

---

## 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-Learn
- Scikit-Image
- Joblib
- tqdm

---

## 📊 Dataset

Dataset used:

**Rock Paper Scissors Dataset**

https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

Dataset contains approximately:

- 712 Paper images
- 726 Rock images
- 750 Scissors images

Total Images: **2188**

---

## ⚙️ Machine Learning Pipeline

1. Load images
2. Resize images to 128×128
3. Convert to grayscale
4. Extract HOG features
5. Normalize features using StandardScaler
6. Train an SVM classifier
7. Save trained model using Joblib
8. Perform real-time predictions from webcam

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/rock-paper-scissors-gesture-recognition.git
```

Move into the project directory

```bash
cd rock-paper-scissors-gesture-recognition
```

Install dependencies

```bash
pip install opencv-python
pip install scikit-learn
pip install scikit-image
pip install numpy
pip install tqdm
pip install joblib
```

---

## ▶️ Train the Model

```bash
python train_model.py
```

After training, the following files will be generated:

```
gesture_model.pkl
scaler.pkl
gesture_names.pkl
```

---

## ▶️ Run the Application

```bash
python predict_webcam.py
```

Place your hand inside the green box.

The application will predict:

- Rock
- Paper
- Scissors

along with a confidence score.

Press **Q** to exit.

---

## 📈 Model Performance

Dataset Size: **2188 images**

Training Accuracy:

```
98.63%
```

Classification Report:

| Class | Precision | Recall | F1-Score |
|--------|-----------|----------|-----------|
| Paper | 0.99 | 0.98 | 0.98 |
| Rock | 0.99 | 0.99 | 0.99 |
| Scissors | 0.99 | 0.99 | 0.99 |

Overall Accuracy:

**98.63%**

---

## Generate the Model

The trained model (`gesture_model.pkl`) is not included in this repository because it exceeds GitHub's recommended file size.

After downloading the dataset, run:

```bash
python train_model.py


## 💡 Future Improvements

- MediaPipe Hand Landmark Integration
- Deep Learning using CNN
- Background-independent hand segmentation
- Gesture smoothing using temporal filtering
- Support for custom gestures
- Deployment as a desktop application

---

## 👨‍💻 Author

**Varshith Raju**

Machine Learning & AI Enthusiast

LinkedIn: *(Add your LinkedIn URL here)*

GitHub: *(Add your GitHub URL here)*

---

## 📄 License

This project is developed for educational and internship purposes.

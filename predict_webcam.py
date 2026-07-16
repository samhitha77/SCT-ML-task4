import cv2
import joblib
import numpy as np
from skimage.feature import hog

# -------------------------
# Load model
# -------------------------
model = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")
gesture_names = joblib.load("gesture_names.pkl")

IMG_SIZE = 128

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# HSV skin range
LOWER = np.array([0, 30, 60], dtype=np.uint8)
UPPER = np.array([20, 170, 255], dtype=np.uint8)

previous_prediction = ""
stable_count = 0
display_prediction = "No Hand"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1 = 180, 80
    x2, y2 = 460, 360

    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER, UPPER)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    confidence = 0

    if contours:

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 5000:

            x, y, w, h = cv2.boundingRect(largest)

            hand = roi[y:y+h, x:x+w]

            if hand.size > 0:

                gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

                feature = hog(
                    gray,
                    orientations=9,
                    pixels_per_cell=(8,8),
                    cells_per_block=(2,2),
                    block_norm="L2-Hys"
                )

                feature = scaler.transform([feature])

                pred = model.predict(feature)[0]
                probs = model.predict_proba(feature)[0]

                confidence = np.max(probs) * 100

                prediction = gesture_names[pred]

                # Stabilize prediction
                if prediction == previous_prediction:
                    stable_count += 1
                else:
                    stable_count = 0

                previous_prediction = prediction

                if stable_count >= 3:
                    display_prediction = prediction

                cv2.rectangle(
                    roi,
                    (x, y),
                    (x+w, y+h),
                    (255,0,0),
                    2
                )

        else:
            display_prediction = "No Hand"

    else:
        display_prediction = "No Hand"

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(
        frame,
        f"Gesture : {display_prediction}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence:.2f}%",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20,430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.imshow("Mask", mask)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
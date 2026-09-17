import cv2
import mediapipe as mp
import pickle
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

with open("gesture_model.pkl", "rb") as file:
    model = pickle.load(file)

print("ML model loaded successfully!")

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access webcam")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    results = detector.detect(mp_image)

    if results.hand_landmarks:

        hand = results.hand_landmarks[0]

        h, w, _ = frame.shape

        features = []

        for landmark in hand:

            x = landmark.x
            y = landmark.y

            features.append(x)
            features.append(y)

            px = int(x * w)
            py = int(y * h)

            cv2.circle(
                frame,
                (px, py),
                5,
                (0, 255, 0),
                -1
            )

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        confidence = np.max(probabilities) * 100

        cv2.putText(
            frame,
            f"Gesture: {prediction}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Real-Time Hand Gesture Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
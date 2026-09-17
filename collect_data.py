import cv2
import mediapipe as mp
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

gesture = input(
    "Enter gesture name (FIST, ONE, TWO, THREE, OPEN_HAND): "
).upper()

filename = "gesture_data.csv"

file_exists = os.path.exists(filename)

file = open(
    filename,
    "a",
    newline=""
)

writer = csv.writer(file)

cap = cv2.VideoCapture(0)

print()
print("Show your gesture to the camera.")
print("Press SPACE to save a sample.")
print("Press Q to quit.")
print()

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

    key = cv2.waitKey(1) & 0xFF

    if results.hand_landmarks:

        hand = results.hand_landmarks[0]

        h, w, _ = frame.shape

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)
        
            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        if key == ord(" "):

            features = []

            for landmark in hand:

                features.append(landmark.x)
                features.append(landmark.y)

            features.append(gesture)
        
            writer.writerow(features)

            print(
                f"Sample saved for {gesture}"
            )
        
        cv2.putText(
            frame,
            "Hand detected",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Save | Q = Quit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Data Collection",
        frame
    )

    if key == ord("q"):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

def get_gesture(points):

    index = points[8][1] < points[6][1]
    middle = points[12][1] < points[10][1]
    ring = points[16][1] < points[14][1]
    little = points[20][1] < points[18][1]

    thumb_up = points[4][1] < points[3][1]
    thumb_down = points[4][1] > points[3][1]

    if thumb_up and not index and not middle and not ring and not little:
        return "THUMBS UP"

    if thumb_down and not index and not middle and not ring and not little:
        return "THUMBS DOWN"

    fingers = 0

    if index:
        fingers += 1

    if middle:
        fingers += 1

    if ring:
        fingers += 1

    if little:
        fingers += 1

    if fingers == 0:
        return "FIST"

    elif fingers == 1:
        return "ONE"

    elif fingers == 2:
        return "TWO"

    elif fingers == 3:
        return "THREE"

    elif fingers == 4:
        return "FOUR"

    elif fingers == 5:
        return "OPEN HAND"

    return "UNKNOWN"

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

        for hand in results.hand_landmarks:

            h, w, _ = frame.shape

            points = []

            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]

            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 255, 255),
                    2
                )

            gesture = get_gesture(points)

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3
            )

    cv2.imshow(
        "Hand Gesture Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
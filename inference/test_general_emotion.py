import cv2
import numpy as np
import time

from features.general_emotion import GeneralEmotionExtractor
from features.behavioral import BehavioralFeatureExtractor


emotion_extractor = GeneralEmotionExtractor()
behavior_extractor = BehavioralFeatureExtractor()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

start_time = time.monotonic()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )

    # Use our existing MediaPipe Tasks face landmarks
    # to obtain the face bounding box.
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    import mediapipe as mp

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = behavior_extractor.detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        h, w, _ = frame.shape

        xs = [
            landmark.x
            for landmark in landmarks
        ]

        ys = [
            landmark.y
            for landmark in landmarks
        ]

        x1 = max(0, int(min(xs) * w) - 20)
        y1 = max(0, int(min(ys) * h) - 20)

        x2 = min(w, int(max(xs) * w) + 20)
        y2 = min(h, int(max(ys) * h) + 20)

        face = frame[y1:y2, x1:x2]

        if face.size > 0:

            probabilities = emotion_extractor.extract(face)

            index = np.argmax(probabilities)

            emotion = emotion_extractor.EMOTIONS[index]

            print(
                " | ".join(
                    f"{name}: {prob:.2f}"
                    for name, prob in zip(
                        emotion_extractor.EMOTIONS,
                        probabilities
                    )
                )
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"General FER: {emotion}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "General Emotion - Face Crop",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

behavior_extractor.close()

cv2.destroyAllWindows()
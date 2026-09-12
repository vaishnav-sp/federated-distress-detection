import cv2
import time

from features.behavioral import BehavioralFeatureExtractor


extractor = BehavioralFeatureExtractor()

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

    features = extractor.extract(
        frame,
        timestamp_ms
    )

    print(features)

    cv2.putText(
        frame,
        f"Movement: {features['movement']:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Eye: {features['eye_openness']:.4f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Mouth: {features['mouth_openness']:.4f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Behavioral Features",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
extractor.close()

cv2.destroyAllWindows()
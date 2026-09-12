import cv2
import time
import numpy as np

from features.fusion import FusedFeatureExtractor


extractor = FusedFeatureExtractor()

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

    print(
        "Feature vector:",
        np.round(features, 4)
    )

    print(
        "Feature dimension:",
        len(features)
    )

    cv2.putText(
        frame,
        f"Features: {len(features)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Fused Emotion + Behavioral Features",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

extractor.close()

cv2.destroyAllWindows()
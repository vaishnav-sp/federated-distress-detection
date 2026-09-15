import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from features.emotion import EmotionFeatureExtractor
from features.general_emotion import GeneralEmotionExtractor


# =========================================================
# Paths
# =========================================================

DATASET_ROOT = "data/raw/Autism emotion recogition dataset"

OUTPUT_FILE = "data/processed/emotion_features.csv"

AUTISM_MODEL = "models/emotion_model.pth"
GENERAL_MODEL = "models/general_fer/model.h5"
FACE_MODEL = "models/face_landmarker.task"


# =========================================================
# Classes
# =========================================================

AUTISM_CLASSES = [
    "anger",
    "fear",
    "joy",
    "Natural",
    "sadness",
    "surprise"
]

GENERAL_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# =========================================================
# Models
# =========================================================

autism_model = EmotionFeatureExtractor(
    AUTISM_MODEL
)

general_model = GeneralEmotionExtractor(
    GENERAL_MODEL
)


# =========================================================
# MediaPipe Face Landmarker
# =========================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(
    model_asset_path=FACE_MODEL
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5
)

detector = vision.FaceLandmarker.create_from_options(
    options
)


# =========================================================
# Output directory
# =========================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


rows = []


# =========================================================
# Face crop
# =========================================================

def get_face_crop(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(
        mp_image
    )

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    h, w, _ = image.shape

    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]

    x1 = max(
        0,
        int(min(xs) * w) - 20
    )

    y1 = max(
        0,
        int(min(ys) * h) - 20
    )

    x2 = min(
        w,
        int(max(xs) * w) + 20
    )

    y2 = min(
        h,
        int(max(ys) * h) + 20
    )

    face = image[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    return face


# =========================================================
# Processing
# =========================================================

total = 0
processed = 0
skipped = 0


for split in ["train", "test"]:

    split_path = os.path.join(
        DATASET_ROOT,
        split
    )

    for class_name in os.listdir(
        split_path
    ):

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.isdir(
            class_path
        ):
            continue

        for filename in os.listdir(
            class_path
        ):

            image_path = os.path.join(
                class_path,
                filename
            )

            image = cv2.imread(
                image_path
            )

            total += 1

            if image is None:

                skipped += 1
                continue

            face = get_face_crop(
                image
            )

            if face is None:

                skipped += 1
                continue

            # =================================================
            # Emotion models
            # =================================================

            autism_probs = autism_model.extract(
                face
            )

            general_probs = general_model.extract(
                face
            )

            # =================================================
            # Row
            # =================================================

            row = {

                "image": image_path,

                "split": split,

                "label": class_name
            }

            # =================================================
            # Autism emotion
            # =================================================

            for i, value in enumerate(
                autism_probs
            ):

                row[
                    f"autism_{AUTISM_CLASSES[i]}"
                ] = float(value)

            # =================================================
            # General emotion
            # =================================================

            for i, value in enumerate(
                general_probs
            ):

                row[
                    f"general_{GENERAL_CLASSES[i]}"
                ] = float(value)

            # =================================================
            # Behavioral features
            #
            # Individual images have no temporal
            # information, therefore these are zero.
            #
            # They will be populated later from video
            # sequences.
            # =================================================

            row["movement"] = 0.0
            row["head_movement"] = 0.0
            row["eye_openness"] = 0.0
            row["mouth_openness"] = 0.0

            row["movement_mean"] = 0.0
            row["movement_std"] = 0.0
            row["movement_peak"] = 0.0
            row["movement_activity"] = 0.0

            row["head_mean"] = 0.0
            row["head_std"] = 0.0
            row["head_peak"] = 0.0

            rows.append(row)

            processed += 1

            if processed % 50 == 0:

                print(
                    f"Processed: "
                    f"{processed}/{total}"
                )


# =========================================================
# Save
# =========================================================

df = pd.DataFrame(
    rows
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


detector.close()

autism_model.close()
general_model.close()


# =========================================================
# Summary
# =========================================================

print(
    "\n=============================="
)

print(
    "FEATURE EXTRACTION COMPLETE"
)

print(
    "=============================="
)

print(
    f"Total images : {total}"
)

print(
    f"Processed    : {processed}"
)

print(
    f"Skipped      : {skipped}"
)

print(
    f"Features     : {len(df.columns) - 3}"
)

print(
    f"Output       : {OUTPUT_FILE}"
)
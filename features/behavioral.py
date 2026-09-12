import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class BehavioralFeatureExtractor:

    def __init__(self, model_path="models/face_landmarker.task"):

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.FaceLandmarker.create_from_options(
            options
        )

        self.previous_landmarks = None

    def extract(self, frame, timestamp_ms):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if not result.face_landmarks:

            self.previous_landmarks = None

            return {
                "face_detected": 0.0,
                "movement": 0.0,
                "head_movement": 0.0,
                "eye_openness": 0.0,
                "mouth_openness": 0.0
            }

        landmarks = result.face_landmarks[0]

        current = np.array([
            [p.x, p.y, p.z]
            for p in landmarks
        ])

        # -----------------------------
        # Overall facial movement
        # -----------------------------

        movement = 0.0

        if self.previous_landmarks is not None:

            movement = np.mean(
                np.linalg.norm(
                    current[:, :2]
                    - self.previous_landmarks[:, :2],
                    axis=1
                )
            )

        # -----------------------------
        # Head movement
        # Use nose position as a
        # lightweight head-motion proxy
        # -----------------------------

        head_movement = 0.0

        if self.previous_landmarks is not None:

            nose_now = current[1, :2]
            nose_previous = self.previous_landmarks[1, :2]

            head_movement = np.linalg.norm(
                nose_now - nose_previous
            )

        self.previous_landmarks = current

        # -----------------------------
        # Eye openness
        # -----------------------------

        left_eye_top = current[159]
        left_eye_bottom = current[145]

        right_eye_top = current[386]
        right_eye_bottom = current[374]

        left_eye = np.linalg.norm(
            left_eye_top[:2] - left_eye_bottom[:2]
        )

        right_eye = np.linalg.norm(
            right_eye_top[:2] - right_eye_bottom[:2]
        )

        eye_openness = (
            left_eye + right_eye
        ) / 2

        # -----------------------------
        # Mouth openness
        # -----------------------------

        mouth_top = current[13]
        mouth_bottom = current[14]

        mouth_openness = np.linalg.norm(
            mouth_top[:2] - mouth_bottom[:2]
        )

        return {
            "face_detected": 1.0,
            "movement": float(movement),
            "head_movement": float(head_movement),
            "eye_openness": float(eye_openness),
            "mouth_openness": float(mouth_openness)
        }

    def extract_from_landmarks(self, landmarks):

        current = np.array([
            [p.x, p.y, p.z]
            for p in landmarks
        ])

        movement = 0.0

        if self.previous_landmarks is not None:

            movement = np.mean(
                np.linalg.norm(
                    current[:, :2]
                    - self.previous_landmarks[:, :2],
                    axis=1
                )
            )

        head_movement = 0.0

        if self.previous_landmarks is not None:

            nose_now = current[1, :2]
            nose_previous = self.previous_landmarks[1, :2]

            head_movement = np.linalg.norm(
                nose_now - nose_previous
            )

        self.previous_landmarks = current

        # Eye openness
        left_eye = np.linalg.norm(
            current[159, :2] -
            current[145, :2]
        )

        right_eye = np.linalg.norm(
            current[386, :2] -
            current[374, :2]
        )

        eye_openness = (
            left_eye + right_eye
        ) / 2

        # Mouth openness
        mouth_openness = np.linalg.norm(
            current[13, :2] -
            current[14, :2]
        )

        return {
            "movement": float(movement),
            "head_movement": float(head_movement),
            "eye_openness": float(eye_openness),
            "mouth_openness": float(mouth_openness)
        }

    def close(self):
        self.detector.close()
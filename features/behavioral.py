import cv2
import numpy as np
import mediapipe as mp

from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class BehavioralFeatureExtractor:

    def __init__(
        self,
        model_path="models/face_landmarker.task",
        history_size=15
    ):

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

        # -----------------------------------------
        # Temporal history
        # -----------------------------------------

        self.history_size = history_size

        self.movement_history = deque(
            maxlen=history_size
        )

        self.head_movement_history = deque(
            maxlen=history_size
        )

        self.eye_history = deque(
            maxlen=history_size
        )

        self.mouth_history = deque(
            maxlen=history_size
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset(self):

        self.previous_landmarks = None

        self.movement_history.clear()
        self.head_movement_history.clear()
        self.eye_history.clear()
        self.mouth_history.clear()

    # =========================================================
    # Extract
    # =========================================================

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

        # -----------------------------------------
        # No face
        # -----------------------------------------

        if not result.face_landmarks:

            self.previous_landmarks = None

            return {
                "face_detected": 0.0,

                "movement": 0.0,
                "head_movement": 0.0,

                "eye_openness": 0.0,
                "mouth_openness": 0.0,

                "movement_mean": 0.0,
                "movement_std": 0.0,
                "movement_peak": 0.0,
                "movement_activity": 0.0,

                "head_mean": 0.0,
                "head_std": 0.0,
                "head_peak": 0.0
            }

        landmarks = result.face_landmarks[0]

        current = np.array(
            [
                [p.x, p.y, p.z]
                for p in landmarks
            ],
            dtype=np.float32
        )

        # =====================================================
        # CURRENT FRAME MOVEMENT
        # =====================================================

        movement = 0.0

        if self.previous_landmarks is not None:

            movement = float(
                np.mean(
                    np.linalg.norm(
                        current[:, :2]
                        - self.previous_landmarks[:, :2],
                        axis=1
                    )
                )
            )

        # =====================================================
        # CURRENT HEAD MOVEMENT
        # =====================================================

        head_movement = 0.0

        if self.previous_landmarks is not None:

            nose_now = current[1, :2]

            nose_previous = (
                self.previous_landmarks[1, :2]
            )

            head_movement = float(
                np.linalg.norm(
                    nose_now - nose_previous
                )
            )

        # Save current landmarks for the NEXT frame.
        self.previous_landmarks = current.copy()

        # =====================================================
        # EYE OPENNESS
        # =====================================================

        left_eye = np.linalg.norm(
            current[159, :2]
            - current[145, :2]
        )

        right_eye = np.linalg.norm(
            current[386, :2]
            - current[374, :2]
        )

        eye_openness = float(
            (left_eye + right_eye) / 2.0
        )

        # =====================================================
        # MOUTH OPENNESS
        # =====================================================

        mouth_openness = float(
            np.linalg.norm(
                current[13, :2]
                - current[14, :2]
            )
        )

        # =====================================================
        # TEMPORAL HISTORY
        # =====================================================

        self.movement_history.append(
            movement
        )

        self.head_movement_history.append(
            head_movement
        )

        self.eye_history.append(
            eye_openness
        )

        self.mouth_history.append(
            mouth_openness
        )

        # =====================================================
        # TEMPORAL FEATURES
        # =====================================================

        movement_array = np.asarray(
            self.movement_history,
            dtype=np.float32
        )

        head_array = np.asarray(
            self.head_movement_history,
            dtype=np.float32
        )

        movement_mean = float(
            np.mean(movement_array)
        )

        movement_std = float(
            np.std(movement_array)
        )

        movement_peak = float(
            np.max(movement_array)
        )

        movement_activity = float(
            np.mean(
                movement_array > 0.015
            )
        )

        head_mean = float(
            np.mean(head_array)
        )

        head_std = float(
            np.std(head_array)
        )

        head_peak = float(
            np.max(head_array)
        )

        # =====================================================
        # RETURN
        # =====================================================

        return {
            "face_detected": 1.0,

            # Current movement only.
            # These are NOT accumulated values.

            "movement": movement,

            "head_movement": head_movement,

            "eye_openness": eye_openness,

            "mouth_openness": mouth_openness,

            # Temporal features retained for the
            # federated 24-D feature representation.

            "movement_mean": movement_mean,
            "movement_std": movement_std,
            "movement_peak": movement_peak,
            "movement_activity": movement_activity,

            "head_mean": head_mean,
            "head_std": head_std,
            "head_peak": head_peak
        }

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        self.detector.close()
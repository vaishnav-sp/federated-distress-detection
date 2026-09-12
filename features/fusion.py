import cv2
import numpy as np

from features.emotion import EmotionFeatureExtractor
from features.general_emotion import GeneralEmotionExtractor
from features.behavioral import BehavioralFeatureExtractor


class FusedFeatureExtractor:

    def __init__(
        self,
        autism_model_path="emotion_model.pth",
        general_model_path="models/general_fer/model.h5",
        landmark_model_path="models/face_landmarker.task"
    ):

        self.autism_emotion = EmotionFeatureExtractor(
            autism_model_path
        )

        self.general_emotion = GeneralEmotionExtractor(
            general_model_path
        )

        self.behavioral = BehavioralFeatureExtractor(
            landmark_model_path
        )

    def get_face_crop(self, frame, landmarks):

        h, w, _ = frame.shape

        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]

        x1 = max(0, int(min(xs) * w) - 20)
        y1 = max(0, int(min(ys) * h) - 20)

        x2 = min(w, int(max(xs) * w) + 20)
        y2 = min(h, int(max(ys) * h) + 20)

        face = frame[y1:y2, x1:x2]

        return face

    def extract(self, frame, timestamp_ms):

        # --------------------------------
        # Detect face using our existing
        # MediaPipe Face Landmarker
        # --------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = __import__("mediapipe").Image(
            image_format=__import__("mediapipe").ImageFormat.SRGB,
            data=rgb
        )

        result = self.behavioral.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # --------------------------------
        # No face detected
        # --------------------------------

        if not result.face_landmarks:

            self.behavioral.previous_landmarks = None

            return None

        landmarks = result.face_landmarks[0]

        face = self.get_face_crop(
            frame,
            landmarks
        )

        if face.size == 0:

            return None

        # --------------------------------
        # Autism-specific emotion
        # --------------------------------

        autism_probs = self.autism_emotion.extract(
            face
        )

        # --------------------------------
        # General FER
        # --------------------------------

        general_probs = self.general_emotion.extract(
            face
        )

        # --------------------------------
        # Behavioral features
        #
        # We already detected the face above.
        # Now calculate behavioral features
        # from the same landmarks.
        # --------------------------------

        behavior = self.behavioral.extract_from_landmarks(
            landmarks
        )

        behavioral_features = np.array([
            behavior["movement"],
            behavior["head_movement"],
            behavior["eye_openness"],
            behavior["mouth_openness"]
        ], dtype=np.float32)

        # --------------------------------
        # Feature fusion
        # --------------------------------

        fused = np.concatenate([
            autism_probs,
            general_probs,
            behavioral_features
        ])

        return fused

    def close(self):

        self.behavioral.close()
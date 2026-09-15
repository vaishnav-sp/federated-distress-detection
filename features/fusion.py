import numpy as np


class FusedFeatureExtractor:

    def __init__(self):

        from features.emotion import EmotionFeatureExtractor
        from features.general_emotion import GeneralEmotionExtractor
        from features.behavioral import BehavioralFeatureExtractor

        # ==========================================
        # Autism-specific emotion model
        # ==========================================

        self.autism_emotion = EmotionFeatureExtractor(
            model_path="models/emotion_model.pth"
        )

        # ==========================================
        # General FER emotion model
        #
        # IMPORTANT:
        # Use GeneralEmotionExtractor here.
        # It contains the correct MobileNetV2
        # preprocessing for the FER model.
        # ==========================================

        self.general_emotion = GeneralEmotionExtractor(
            model_path="models/general_fer/model.h5"
        )

        # ==========================================
        # Behavioral feature extractor
        # ==========================================

        self.behavioral = BehavioralFeatureExtractor()


    def extract(self, frame, timestamp_ms):

        # ==========================================
        # 1. Autism emotion features
        # ==========================================

        autism_features = self.autism_emotion.extract(
            frame
        )

        autism = np.asarray(
            autism_features,
            dtype=np.float32
        )


        # ==========================================
        # 2. General emotion features
        # ==========================================

        general_features = self.general_emotion.extract(
            frame
        )

        general = np.asarray(
            general_features,
            dtype=np.float32
        )


        # ==========================================
        # 3. Behavioral features
        # ==========================================

        behavioral = self.behavioral.extract(
            frame,
            timestamp_ms
        )


        # ==========================================
        # Current behavioral features
        # ==========================================

        behavior_current = np.array([
            behavioral["movement"],
            behavioral["head_movement"],
            behavioral["eye_openness"],
            behavioral["mouth_openness"]
        ], dtype=np.float32)


        # ==========================================
        # Temporal behavioral features
        # ==========================================

        behavior_temporal = np.array([
            behavioral["movement_mean"],
            behavioral["movement_std"],
            behavioral["movement_peak"],
            behavioral["movement_activity"],

            behavioral["head_mean"],
            behavioral["head_std"],
            behavioral["head_peak"]
        ], dtype=np.float32)


        # ==========================================
        # Final feature vector
        #
        # 6  autism emotion
        # 7  general emotion
        # 4  current behavior
        # 7  temporal behavior
        #
        # TOTAL = 24
        # ==========================================

        features = np.concatenate([
            autism,
            general,
            behavior_current,
            behavior_temporal
        ])


        return features.astype(
            np.float32
        )


    def close(self):

        self.autism_emotion.close()
        self.general_emotion.close()
        self.behavioral.close()
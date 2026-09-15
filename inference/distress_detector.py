from collections import deque
import numpy as np


class DistressDetector:

    def __init__(self, window_size=30):

        self.window = deque(maxlen=window_size)

    def update(self, features):

        if features is None:
            return {
                "label": "NO FACE",
                "score": 0.0
            }

        self.window.append(features)

        # Need enough frames to establish a baseline
        if len(self.window) < 10:
            return {
                "label": "NORMAL",
                "score": 0.0
            }

        data = np.array(self.window)

        # ==========================================
        # 1. EMOTION RISK
        # ==========================================

        # Autism model
        autism_negative = (
            0.30 * data[:, 0] +   # anger
            0.30 * data[:, 1] +   # fear
            0.30 * data[:, 4]     # sadness
        )

        # General FER
        general_negative = (
            0.25 * data[:, 6] +   # angry
            0.15 * data[:, 7] +   # disgust
            0.25 * data[:, 8] +   # fear
            0.25 * data[:, 11]    # sad
        )

        emotion_risk = (
            autism_negative +
            general_negative
        )

        emotion_score = float(
            np.mean(emotion_risk)
        )

        # ==========================================
        # 2. MOVEMENT FEATURES
        # ==========================================

        movement = data[:, 13]
        head_movement = data[:, 14]

        # Recent movement
        recent_n = min(8, len(data))

        recent_movement = movement[-recent_n:]
        recent_head = head_movement[-recent_n:]

        # ==========================================
        # 3. ESTABLISH BASELINE
        # ==========================================

        baseline_n = max(
            5,
            len(data) // 2
        )

        baseline_movement = movement[:-recent_n]

        if len(baseline_movement) < 5:
            baseline_movement = movement[:baseline_n]

        movement_baseline = np.mean(
            baseline_movement
        )

        movement_std = np.std(
            baseline_movement
        )

        head_baseline = np.mean(
            head_movement[:-recent_n]
        )

        head_std = np.std(
            head_movement[:-recent_n]
        )

        # Avoid zero variance
        movement_threshold = max(
            movement_baseline + 2 * movement_std,
            0.02
        )

        head_threshold = max(
            head_baseline + 2 * head_std,
            0.01
        )

        # ==========================================
        # 4. MOVEMENT ANOMALY
        # ==========================================

        movement_anomalies = (
            recent_movement >
            movement_threshold
        )

        head_anomalies = (
            recent_head >
            head_threshold
        )

        movement_anomaly_ratio = np.mean(
            movement_anomalies
        )

        head_anomaly_ratio = np.mean(
            head_anomalies
        )

        # ==========================================
        # 5. MOVEMENT INTENSITY
        # ==========================================

        movement_intensity = np.mean(
            recent_movement
        ) / (
            movement_baseline + 1e-6
        )

        head_intensity = np.mean(
            recent_head
        ) / (
            head_baseline + 1e-6
        )

        movement_intensity = min(
            movement_intensity / 5.0,
            1.0
        )

        head_intensity = min(
            head_intensity / 5.0,
            1.0
        )

        # ==========================================
        # 6. REPEATED MOVEMENT
        # ==========================================

        repeated_movement = (
            0.6 * movement_anomaly_ratio +
            0.4 * head_anomaly_ratio
        )

        # ==========================================
        # 7. FINAL BEHAVIOR SCORE
        # ==========================================

        behavior_score = (
            0.45 * movement_anomaly_ratio +
            0.25 * head_anomaly_ratio +
            0.20 * movement_intensity +
            0.10 * head_intensity
        )

        # ==========================================
        # 8. RECENT NEGATIVE EMOTION
        # ==========================================

        recent_emotion = np.mean(
            emotion_risk[-recent_n:]
        )

        # ==========================================
        # 9. FINAL DISTRESS SCORE
        # ==========================================

        # Emotion and behavior are independent
        # evidence sources.

        distress_score = (
            0.45 * emotion_score +
            0.40 * behavior_score +
            0.15 * recent_emotion
        )

        # ==========================================
        # 10. IMPORTANT BEHAVIOR OVERRIDE
        # ==========================================

        # Strong abnormal/repeated movement can
        # indicate potential distress even when
        # facial emotion is relatively neutral.

        if (
            movement_anomaly_ratio >= 0.60
            and behavior_score >= 0.45
        ):
            distress_score = max(
                distress_score,
                0.45
            )

        # ==========================================
        # 11. FINAL DECISION
        # ==========================================

        if distress_score >= 0.30:

            label = "POTENTIAL DISTRESS"

        else:

            label = "NORMAL"

        return {
            "label": label,
            "score": float(
                min(distress_score, 1.0)
            ),
            "emotion_score": float(
                min(emotion_score, 1.0)
            ),
            "behavior_score": float(
                min(behavior_score, 1.0)
            ),
            "movement_anomaly": float(
                movement_anomaly_ratio
            ),
            "repeated_movement": float(
                repeated_movement
            )
        }
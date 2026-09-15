import numpy as np


class DistressScorer:

    def __init__(self):

        # =========================================
        # GENERAL FER EMOTION INDICES
        # =========================================

        self.general_angry = 6
        self.general_disgust = 7
        self.general_fear = 8
        self.general_happy = 9
        self.general_neutral = 10
        self.general_sad = 11
        self.general_surprise = 12

        # =========================================
        # BEHAVIOR INDICES
        # =========================================

        self.movement = 13
        self.head_movement = 14

        # =========================================
        # DISTRESS THRESHOLDS
        # =========================================

        # Displayed movement score is 0-100.
        #
        # Movement must reach 25 before it can
        # independently trigger distress.
        self.movement_threshold = 25.0

        # Negative emotion must have at least
        # this probability to trigger distress.
        self.negative_emotion_threshold = 0.35

    def calculate(self, features):

        features = np.asarray(
            features,
            dtype=np.float32
        )

        # =========================================
        # 1. GENERAL EMOTION
        # =========================================

        angry = float(
            features[self.general_angry]
        )

        disgust = float(
            features[self.general_disgust]
        )

        fear = float(
            features[self.general_fear]
        )

        happy = float(
            features[self.general_happy]
        )

        neutral = float(
            features[self.general_neutral]
        )

        sad = float(
            features[self.general_sad]
        )

        surprise = float(
            features[self.general_surprise]
        )

        # =========================================
        # 2. DETECT STRONGEST EMOTION
        # =========================================

        emotion_values = {
            "angry": angry,
            "disgust": disgust,
            "fear": fear,
            "happy": happy,
            "neutral": neutral,
            "sad": sad,
            "surprise": surprise
        }

        detected_emotion = max(
            emotion_values,
            key=emotion_values.get
        )

        emotion_confidence = float(
            emotion_values[detected_emotion]
        )

        # =========================================
        # 3. NEGATIVE EMOTION
        # =========================================

        negative_emotions = {
            "angry",
            "disgust",
            "fear",
            "sad"
        }

        negative_emotion = (
            detected_emotion in negative_emotions
            and
            emotion_confidence
            >= self.negative_emotion_threshold
        )

        # =========================================
        # 4. RAW MOVEMENT
        # =========================================

        raw_movement = float(
            features[self.movement]
        )

        raw_head_movement = float(
            features[self.head_movement]
        )

        # =========================================
        # 5. MOVEMENT SCORE 0-100
        # =========================================
        #
        # Raw MediaPipe movement is normally a small
        # value such as:
        #
        # 0.005
        # 0.010
        # 0.020
        # 0.030
        #
        # Convert it to a readable 0-100 score.
        #
        # 0.010 -> 10
        # 0.020 -> 20
        # 0.025 -> 25
        # 0.030 -> 30
        #
        # This makes the threshold easy to understand.
        # =========================================

        movement_score = raw_movement * 1000.0

        head_score = raw_head_movement * 1000.0

        # =========================================
        # 6. COMBINED MOVEMENT
        # =========================================

        movement_score = (
            0.7 * movement_score
            + 0.3 * head_score
        )

        movement_score = float(
            np.clip(
                movement_score,
                0.0,
                100.0
            )
        )

        # =========================================
        # 7. MOVEMENT TRIGGER
        # =========================================

        high_movement = (
            movement_score
            >= self.movement_threshold
        )

        # =========================================
        # 8. DISTRESS DECISION
        # =========================================
        #
        # RULE 1:
        #
        # Strong negative emotion
        # -> distress
        #
        # RULE 2:
        #
        # Movement >= 25
        # -> distress
        #
        # RULE 3:
        #
        # Happy/neutral + movement < 25
        # -> normal
        #
        # There is NO previous-state memory.
        # =========================================

        if negative_emotion:

            label = "POTENTIAL DISTRESS"

            reason = (
                f"negative emotion: "
                f"{detected_emotion}"
            )

        elif high_movement:

            label = "POTENTIAL DISTRESS"

            reason = "high movement"

        else:

            label = "NORMAL"

            reason = "no distress trigger"

        # =========================================
        # 9. EMOTION SCORE
        # =========================================
        #
        # Only general FER negative emotions.
        # =========================================

        emotion_score = max(
            angry,
            disgust,
            fear,
            sad
        )

        emotion_score = float(
            np.clip(
                emotion_score,
                0.0,
                1.0
            )
        )

        # =========================================
        # 10. RETURN RESULT
        # =========================================

        return {
            "emotion": detected_emotion,

            "emotion_confidence": emotion_confidence,

            "emotion_score": emotion_score,

            "movement_score": movement_score,

            "raw_movement": raw_movement,

            "head_movement": raw_head_movement,

            "negative_emotion": bool(
                negative_emotion
            ),

            "high_movement": bool(
                high_movement
            ),

            "distress_score": float(
                max(
                    emotion_score,
                    movement_score / 100.0
                )
            ),

            "label": label,

            "reason": reason
        }
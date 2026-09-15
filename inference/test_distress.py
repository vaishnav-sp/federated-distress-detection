import cv2
import time
import numpy as np

from features.emotion import EmotionFeatureExtractor
from features.behavioral import BehavioralFeatureExtractor


# ============================================================
# CONFIGURATION
# ============================================================

AUTISM_MODEL = "models/emotion_model.pth"
GENERAL_MODEL = "models/general_fer/model.h5"

# A fresh decision is made every 5 seconds.
WINDOW_SECONDS = 5.0

# Number of recent frames used only for displaying a stable
# movement score.
MOVEMENT_HISTORY = 30

# Raw MediaPipe landmark movement considered "high".
MOVEMENT_THRESHOLD = 0.035

# At least this fraction of the 5-second window must contain
# high movement for movement-only distress.
HIGH_MOVEMENT_RATIO = 0.35

# IMPORTANT:
# We use ONLY the general FER model for distress decisions.
# A negative emotion must be reasonably confident.
NEGATIVE_EMOTION_THRESHOLD = 0.40

# A negative emotion must persist through this fraction of
# the 5-second window before it triggers distress.
#
# This prevents one or two incorrect "sad/angry" predictions
# from making the whole window distress.
NEGATIVE_EMOTION_RATIO = 0.40


# ============================================================
# EMOTION LABELS
# ============================================================

GENERAL_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

NEGATIVE_EMOTIONS = {
    "angry",
    "disgust",
    "fear",
    "sad"
}


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("==============================")
print("LOADING DISTRESS DETECTOR")
print("==============================")

autism_emotion = EmotionFeatureExtractor(
    model_path=AUTISM_MODEL
)

# DO NOT CHANGE THE GENERAL EMOTION MODEL/PREPROCESSING.
general_emotion = EmotionFeatureExtractor(
    model_path=GENERAL_MODEL
)

behavioral = BehavioralFeatureExtractor()

print("Models loaded successfully.")
print()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")

    autism_emotion.close()
    general_emotion.close()
    behavioral.close()

    raise SystemExit


# ============================================================
# STATE
# ============================================================

start_time = time.time()
window_start = time.time()

movement_history = []

current_label = "NORMAL"

last_general_emotion = "neutral"
last_general_confidence = 0.0

last_autism_emotion = "Natural"
last_autism_confidence = 0.0

current_movement = 0.0
movement_score = 0.0

window_negative_frames = 0
window_high_movement_frames = 0
window_total_frames = 0

last_negative_ratio = 0.0
last_movement_ratio = 0.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_emotion(probabilities, labels):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    index = int(
        np.argmax(probabilities)
    )

    return (
        labels[index],
        float(probabilities[index])
    )


def calculate_movement_score(history):

    if len(history) < 2:
        return 0.0

    values = np.asarray(
        history,
        dtype=np.float32
    )

    mean_movement = float(
        np.mean(values)
    )

    # 0.035 raw movement ~= 0.5 score
    # 0.070 raw movement ~= 1.0 score
    score = mean_movement / (
        MOVEMENT_THRESHOLD * 2.0
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


def reset_window():

    return 0, 0, 0, time.time()


# ============================================================
# START
# ============================================================

print("==============================")
print("STARTING REAL-TIME DETECTION")
print("==============================")
print()
print("GENERAL FER IS USED FOR DISTRESS DECISION")
print()
print("Negative emotion -> distress")
print("High movement + non-negative emotion -> distress")
print("Stable positive/neutral + low movement -> normal")
print()
print("Observation window:", WINDOW_SECONDS, "seconds")
print("Negative confidence:", NEGATIVE_EMOTION_THRESHOLD)
print("Negative ratio:", NEGATIVE_EMOTION_RATIO)
print("High movement ratio:", HIGH_MOVEMENT_RATIO)
print()
print("Press Q to quit.")
print()


try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Failed to read frame.")
            break

        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        # ====================================================
        # 1. GENERAL EMOTION
        #
        # THIS PART IS LEFT AS-IS.
        # ====================================================

        general_probs = general_emotion.extract(
            frame
        )

        general_label, general_confidence = get_emotion(
            general_probs,
            GENERAL_EMOTIONS
        )

        last_general_emotion = general_label
        last_general_confidence = general_confidence

        # ====================================================
        # 2. AUTISM EMOTION
        #
        # Display only.
        # NOT USED FOR DISTRESS DECISION.
        # ====================================================

        autism_probs = autism_emotion.extract(
            frame
        )

        autism_labels = [
            "anger",
            "fear",
            "joy",
            "Natural",
            "sadness",
            "surprise"
        ]

        autism_label, autism_confidence = get_emotion(
            autism_probs,
            autism_labels
        )

        last_autism_emotion = autism_label
        last_autism_confidence = autism_confidence

        # ====================================================
        # 3. BEHAVIOR / MOVEMENT
        # ====================================================

        behavior = behavioral.extract(
            frame,
            timestamp_ms
        )

        current_movement = float(
            behavior.get(
                "movement",
                0.0
            )
        )

        movement_history.append(
            current_movement
        )

        if len(movement_history) > MOVEMENT_HISTORY:

            movement_history.pop(0)

        movement_score = calculate_movement_score(
            movement_history
        )

        # ====================================================
        # 4. GENERAL EMOTION NEGATIVE CHECK
        #
        # IMPORTANT:
        # Autism emotion is NOT checked here.
        #
        # Example:
        # Autism = fear 0.80
        # General = happy 0.60
        #
        # This does NOT become distress because of autism fear.
        # ====================================================

        general_negative = (
            general_label in NEGATIVE_EMOTIONS
            and
            general_confidence >= NEGATIVE_EMOTION_THRESHOLD
        )

        # ====================================================
        # 5. HIGH MOVEMENT CHECK
        # ====================================================

        high_movement = (
            current_movement >= MOVEMENT_THRESHOLD
        )

        # ====================================================
        # 6. ADD CURRENT FRAME TO 5-SECOND WINDOW
        # ====================================================

        window_total_frames += 1

        if general_negative:

            window_negative_frames += 1

        if high_movement:

            window_high_movement_frames += 1

        # ====================================================
        # 7. CHECK WINDOW
        # ====================================================

        now = time.time()

        window_elapsed = (
            now - window_start
        )

        if window_elapsed >= WINDOW_SECONDS:

            total = max(
                window_total_frames,
                1
            )

            negative_ratio = (
                window_negative_frames
                / total
            )

            movement_ratio = (
                window_high_movement_frames
                / total
            )

            last_negative_ratio = negative_ratio
            last_movement_ratio = movement_ratio

            # -----------------------------------------------
            # CONDITION A: SUSTAINED NEGATIVE EMOTION
            #
            # A clearly sad/angry/fearful face that remains
            # negative for most of the observation window
            # can trigger distress even when completely still.
            # -----------------------------------------------

            negative_distress = (
                negative_ratio >= NEGATIVE_EMOTION_RATIO
            )

            # -----------------------------------------------
            # CONDITION B: HIGH MOVEMENT
            #
            # Movement alone can trigger distress.
            # This is intentionally based on the fraction of
            # frames with genuinely high raw movement.
            # -----------------------------------------------

            movement_distress = (
                movement_ratio >= HIGH_MOVEMENT_RATIO
            )

            # -----------------------------------------------
            # FRESH DECISION
            #
            # There is NO previous-state memory here.
            # The result of this window completely replaces
            # the result of the previous window.
            # -----------------------------------------------

            if negative_distress:

                current_label = "POTENTIAL DISTRESS"

            elif movement_distress:

                current_label = "POTENTIAL DISTRESS"

            else:

                current_label = "NORMAL"

            # -----------------------------------------------
            # RESET THE WINDOW COMPLETELY
            # -----------------------------------------------

            (
                window_negative_frames,
                window_high_movement_frames,
                window_total_frames,
                window_start
            ) = reset_window()

        # ====================================================
        # 8. TERMINAL OUTPUT
        # ====================================================

        print(
            f"\r"
            f"Emotion: {general_label:<8} "
            f"({general_confidence:.2f}) | "
            f"Autism: {autism_label:<8} "
            f"({autism_confidence:.2f}) | "
            f"Movement: {movement_score:.2f} | "
            f"Window Neg: {last_negative_ratio:.2f} | "
            f"Window Move: {last_movement_ratio:.2f} | "
            f"{current_label:<18}",
            end=""
        )

        # ====================================================
        # 9. CAMERA DISPLAY
        # ====================================================

        if current_label == "POTENTIAL DISTRESS":

            label_color = (
                0,
                0,
                255
            )

        else:

            label_color = (
                0,
                255,
                0
            )

        cv2.putText(
            frame,
            current_label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            label_color,
            2
        )

        cv2.putText(
            frame,
            f"General: {general_label} "
            f"({general_confidence:.2f})",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Autism: {autism_label} "
            f"({autism_confidence:.2f})",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Movement: {movement_score:.2f}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Negative ratio: {last_negative_ratio:.2f}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Movement ratio: {last_movement_ratio:.2f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Decision window: 5 sec",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Federated Distress Detection",
            frame
        )

        # ====================================================
        # 10. QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


finally:

    cap.release()

    cv2.destroyAllWindows()

    autism_emotion.close()
    general_emotion.close()
    behavioral.close()

    print()
    print()
    print("==============================")
    print("DETECTION STOPPED")
    print("==============================")

import numpy as np
import tensorflow as tf


class GeneralEmotionExtractor:

    EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ]

    def __init__(self, model_path="models/general_fer/model.h5"):

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    def extract(self, frame):

        # BGR -> RGB
        frame = frame[:, :, ::-1]

        # Model input
        image = tf.image.resize(
            frame,
            (224, 224)
        )

        image = tf.cast(
            image,
            tf.float32
        )

        # MobileNetV2 preprocessing
        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        image = tf.expand_dims(
            image,
            axis=0
        )

        probabilities = self.model.predict(
            image,
            verbose=0
        )[0]

        return probabilities
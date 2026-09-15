import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class EmotionFeatureExtractor:

    def __init__(self, model_path="models/emotion_model.pth"):

        self.model_path = model_path

        # =====================================================
        # Determine model type
        # =====================================================

        if model_path.lower().endswith(".h5"):

            # ---------------------------------------------
            # TensorFlow / Keras general emotion model
            # ---------------------------------------------

            import tensorflow as tf

            self.model_type = "tensorflow"

            self.device = None

            self.model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

            self.transform = None

        else:

            # ---------------------------------------------
            # PyTorch autism emotion model
            # ---------------------------------------------

            from models.emotion_model import EmotionModel

            self.model_type = "pytorch"

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            self.model = EmotionModel(
                num_classes=6
            )

            self.model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=False
                )
            )

            self.model.to(self.device)

            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225
                    ]
                )
            ])

    # =========================================================
    # Feature extraction
    # =========================================================

    def extract(self, frame):

        # =====================================================
        # PyTorch model
        # =====================================================

        if self.model_type == "pytorch":

            image = self.transform(frame)

            image = image.unsqueeze(0).to(
                self.device
            )

            with torch.no_grad():

                logits = self.model(
                    image
                )

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

            return (
                probabilities
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        # =====================================================
        # TensorFlow model
        # =====================================================

        else:

            # OpenCV frame is BGR.
            # Convert to RGB.

            rgb = frame[:, :, ::-1]

            image =rgb.astype(
                np.float32
            )

            # Resize to model input size

            import cv2

            image = cv2.resize(
                image,
                (224, 224)
            )

            image = image / 255.0

            image = np.expand_dims(
                image,
                axis=0
            )

            probabilities = self.model.predict(
                image,
                verbose=0
            )

            probabilities = np.asarray(
                probabilities[0],
                dtype=np.float32
            )

            return probabilities

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        # PyTorch model doesn't require explicit cleanup.
        # TensorFlow model also doesn't require explicit close.

        pass
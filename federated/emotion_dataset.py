import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FederatedEmotionDataset(Dataset):

    # Original emotion -> binary distress label
    #
    # 0 = NORMAL
    # 1 = DISTRESS

    CLASS_MAPPING = {
        "joy": 0,
        "Natural": 0,
        "surprise": 0,

        "anger": 1,
        "fear": 1,
        "sadness": 1,
    }

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None
    ):

        self.root_dir = Path(root_dir)
        self.split = split

        self.data_dir = (
            self.root_dir / split
        )

        if not self.data_dir.exists():

            raise RuntimeError(
                f"Dataset directory not found: "
                f"{self.data_dir}"
            )

        self.transform = transform

        if self.transform is None:

            self.transform = transforms.Compose([
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

        self.samples = []

        self._load_samples()

    # ========================================================
    # LOAD DATA
    # ========================================================

    def _load_samples(self):

        for emotion, label in self.CLASS_MAPPING.items():

            emotion_dir = (
                self.data_dir / emotion
            )

            if not emotion_dir.exists():

                print(
                    f"WARNING: Folder not found: "
                    f"{emotion_dir}"
                )

                continue

            for file in emotion_dir.rglob("*"):

                if file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                self.samples.append(
                    (
                        str(file),
                        label,
                        emotion
                    )
                )

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No images found in dataset: "
                f"{self.data_dir}"
            )

    # ========================================================
    # DATASET INTERFACE
    # ========================================================

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        path, label, emotion = self.samples[index]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image, label

    # ========================================================
    # INFORMATION
    # ========================================================

    def get_class_distribution(self):

        distribution = {
            "NORMAL": 0,
            "DISTRESS": 0
        }

        for _, label, _ in self.samples:

            if label == 0:
                distribution["NORMAL"] += 1

            else:
                distribution["DISTRESS"] += 1

        return distribution

    def get_original_distribution(self):

        distribution = {}

        for _, _, emotion in self.samples:

            if emotion not in distribution:

                distribution[emotion] = 0

            distribution[emotion] += 1

        return distribution
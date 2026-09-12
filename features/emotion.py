import torch
import torch.nn.functional as F
from torchvision import transforms

from models.emotion_model import EmotionModel


class EmotionFeatureExtractor:

    def __init__(self, model_path="emotion_model.pth"):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = EmotionModel(num_classes=6)

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, frame):

        image = self.transform(frame)

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():

            logits = self.model(image)

            probabilities = F.softmax(
                logits,
                dim=1
            )

        return probabilities.squeeze(0).cpu().numpy()
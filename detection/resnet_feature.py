import torch

from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from detection.config import (
    DEVICE,
    FRAME_WIDTH,
    FRAME_HEIGHT
)


class ResNetFeatureExtractor:

    def __init__(self):

        self.model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        self.model.fc = torch.nn.Identity()

        self.model.eval()

        self.model.to(DEVICE)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (FRAME_HEIGHT, FRAME_WIDTH)
            ),
            transforms.ToTensor()
        ])

        print("✓ ResNet50 Feature Extractor Ready")

    def extract(self, image):

        image = self.transform(image)

        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            feature = self.model(image)

        return (
            feature
            .cpu()
            .numpy()
            .flatten()
        )
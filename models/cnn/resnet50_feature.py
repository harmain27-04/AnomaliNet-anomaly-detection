"""
=========================================================
AnomaliNet ResNet50 Feature Extraction Module
=========================================================
Extracts 2048-dimensional spatial feature vectors
from detected person regions.
=========================================================
"""

import torch
from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# LOAD RESNET50
# =====================================================

resnet = resnet50(
    weights=ResNet50_Weights.DEFAULT
)

# Remove classification layer
resnet.fc = torch.nn.Identity()

resnet.eval()
resnet.to(device)

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(image):

    """
    Extracts a 2048-dimensional feature vector
    from an OpenCV image (NumPy array).

    Parameters
    ----------
    image : ndarray
        ROI obtained from YOLO detection.

    Returns
    -------
    numpy.ndarray
        Feature vector of size (2048,)
    """

    if image is None or image.size == 0:
        return None

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        features = resnet(image_tensor)

    features = features.squeeze(0)

    features = features.cpu().numpy()

    features = features.astype("float32")

    return features
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionProcessingModel(nn.Module):
    def __init__(self, model_name="resnet50", num_classes=1000, top_k=5):
        super(VisionProcessingModel, self).__init__()
        
        # Load pre-trained model
        self.model = self._load_pretrained_model(model_name, num_classes)
        self.model.eval()  # Set the model to evaluation mode

        # Automatically detect device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

        # Number of top predictions to return
        self.top_k = top_k

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_pretrained_model(self, model_name, num_classes):
        """Helper function to load a pre-trained model."""
        if model_name == "resnet50":
            return models.resnet50(pretrained=True)
        elif model_name == "efficientnet_b0":
            return models.efficientnet_b0(pretrained=True)
        elif model_name == "mobilenet_v2":
            return models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, image):
        """Forward pass through the model."""
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        return outputs

    def process_image(self, image_path):
        """Process an image and get predictions."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Apply transformations and add batch dimension
            image = self.transform(image).unsqueeze(0)

            # Get predictions from model
            outputs = self.forward(image)

            # Convert raw output to probabilities (Softmax)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Get top-k categories
            top_k_prob, top_k_catid = torch.topk(probabilities, self.top_k)

            return top_k_prob, top_k_catid
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None, None

    def get_category_labels(self, category_ids):
        """Map category IDs to human-readable labels."""
        # Load ImageNet class labels
        labels_path = os.getenv("IMAGENET_LABELS_PATH", "path/to/imagenet_labels.txt")
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]

        # Map category IDs to labels
        return [labels[cat_id] for cat_id in category_ids]

    def enhance_vision_processing(self, image_path):
        """Enhance vision capabilities by extracting top-k predictions."""
        top_k_prob, top_k_catid = self.process_image(image_path)

        if top_k_prob is not None and top_k_catid is not None:
            # Convert tensors to lists
            top_k_prob = top_k_prob.tolist()
            top_k_catid = top_k_catid.tolist()

            # Get human-readable labels
            category_labels = self.get_category_labels(top_k_catid)

            return top_k_prob, top_k_catid, category_labels
        else:
            return None, None, None

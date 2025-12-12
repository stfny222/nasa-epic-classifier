"""
Image Preprocessing Transforms
===============================

Standard transforms for image preprocessing shared between training and inference.
Ensures consistency between training-time and inference-time transformations.
"""

from torchvision import transforms

# Inference transform (used by server and validation)
# No augmentation - deterministic preprocessing only
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Training transform (includes augmentation)
# Applied to training data only
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation transform (same as inference - no augmentation)
VAL_TRANSFORM = INFERENCE_TRANSFORM

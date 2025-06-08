import argparse
import torch
import json
from torchvision import models
from PIL import Image
import numpy as np

def load_checkpoint(filepath):
    """Loads a model checkpoint and rebuilds the model."""
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # Ensure architecture exists
    arch = checkpoint.get('arch', 'vgg16')
    
    if arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif arch == 'vgg13':
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported architecture '{arch}' found in checkpoint.")

    # Load classifier and state_dict
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    """Processes an image into a tensor format suitable for the model."""
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

    # Resize while maintaining aspect ratio
    pil_image.thumbnail((256, 256))

    # Crop the center 224x224
    left = (pil_image.width - 224) / 2
    top = (pil_image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    pil_image = pil_image.crop((left, top, right, bottom))

    # Normalize
    np_image = np.array(pil_image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions for PyTorch
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).float()

def predict(image_path, model, topk=5):
    """Predicts the class of an image using a trained deep learning model."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess image
    image = process_image(image_path).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)

    # Get the top K probabilities and indices
    top_p, top_indices = probabilities.topk(topk, dim=1)
    top_p = top_p.cpu().numpy().squeeze()
    top_indices = top_indices.cpu().numpy().squeeze()

    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]

    return top_p, top_classes

def main():
    parser = argparse.ArgumentParser(description="Predict image class with trained model.")
    
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Top K classes to return')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Load model
    model = load_checkpoint(args.checkpoint)

    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Make prediction
    probs, classes = predict(args.image_path, model, args.top_k)

    # Map to category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:  # FIXED INDENTATION
            cat_to_name = json.load(f)

        # Convert class indices to actual labels before using category names
        classes = [cat_to_name.get(str(cls), "Unknown") for cls in classes]

    # Print results
    print(f"Top {args.top_k} Classes:")
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob:.2f}")

if __name__ == '__main__':
    main()
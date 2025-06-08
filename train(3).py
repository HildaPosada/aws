import argparse
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

def load_data(data_dir):
    """Load the dataset and apply transformations."""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=64, shuffle=False),
        'test': DataLoader(image_datasets['test'], batch_size=64, shuffle=False)
    }

    return dataloaders, image_datasets['train'].class_to_idx

def build_model(arch='vgg16', hidden_units=512, learning_rate=0.003):
    """Builds a deep learning model with a pre-trained architecture."""
    supported_models = {
        'vgg16': models.vgg16,
        'vgg13': models.vgg13,
        'resnet18': models.resnet18,
        'densenet121': models.densenet121,
        'alexnet': models.alexnet,
    }

    if arch not in supported_models:
        print(f"Error: {arch} is not a supported architecture. Please choose from: {', '.join(supported_models.keys())}.")
        return None, None, None

    # Load the chosen model with pre-trained weights
    model = supported_models[arch](weights="IMAGENET1K_V1")

    # Determine input size dynamically
    if hasattr(model, 'fc'):  # For ResNet and similar architectures
        input_size = model.fc.in_features
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # For VGG and DenseNet
        input_size = model.classifier[0].in_features
    else:
        print("Error: Unable to determine input size for the classifier.")
        return None, None, None

    # Freeze pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),  # 102 flower classes
        nn.LogSoftmax(dim=1)
    )

    # Attach classifier to model
    if hasattr(model, 'fc'):
        model.fc = classifier
    else:
        model.classifier = classifier

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def save_checkpoint(model, optimizer, save_dir, class_to_idx, arch, epochs):
    """Saves the trained model checkpoint."""
    checkpoint = {
        'arch': arch,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'classifier': model.classifier,
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"Checkpoint saved at {save_dir}/checkpoint.pth")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a new deep learning model')

    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='./', help='Checkpoint save directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg13, vgg16, resnet18, densenet121, alexnet)')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    print(f"Using PyTorch version: {torch.__version__}")
    print(f"GPU Available: {torch.cuda.is_available()}")

    # Load data
    dataloaders, class_to_idx = load_data(args.data_dir)

    # Build model
    model, criterion, optimizer = build_model(args.arch, args.hidden_units, args.learning_rate)
    if model is None:
        print("Model initialization failed. Exiting...")
        return

    # Use GPU if available
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    print(f"Training model with {args.epochs} epochs...")
    steps = 0
    running_loss = 0
    print_every = 40

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        validation_loss += criterion(logps, labels).item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

    save_checkpoint(model, optimizer, args.save_dir, class_to_idx, args.arch, args.epochs)
    print("Model training complete. Checkpoint saved.")

if __name__ == '__main__':
    main()

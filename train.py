import argparse
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from os.path import exists

# Argument parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description="Neural Network Training Script")
    parser.add_argument('--architecture', type=str, default="vgg16", help='Model architecture')
    parser.add_argument('--checkpoint_dir', default="./model_checkpoint.pth", help='Checkpoint directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--units', type=int, default=120, help='Hidden units')
    parser.add_argument('--training_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--use_gpu', type=str, default="gpu", help='GPU usage flag')
    return parser.parse_args()

# Image data transformation for training
def training_data_transform(training_directory):
    transformations = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(training_directory, transform=transformations)
    return dataset

# Image data transformation for testing
def testing_data_transform(testing_directory):
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(testing_directory, transform=transformations)
    return dataset

# Data loader function
def load_data(dataset, is_train=True):
    loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=is_train)
    return loader

# GPU availability check
def gpu_check(use_gpu):
    if not use_gpu:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Using CPU as CUDA is not available.")
    return device

# Model loader based on architecture
def load_model(arch="vgg16"):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    for param in model.parameters():
        param.requires_grad = False
    return model

# Initialize the classifier
def setup_classifier(model, units):
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(units, 90)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(90, 70)),
        ('relu3', nn.ReLU()),
        ('fc4', nn.Linear(70, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model

# Validation process
def validate(model, validation_loader, criterion, gpu):
    test_loss = 0
    accuracy = 0
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(gpu), labels.to(gpu)
        output = model(inputs)
        test_loss += criterion(output, labels).item()
        probabilities = torch.exp(output)
        top_p, top_class = probabilities.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return test_loss, accuracy

# Network training function
def train_network(model, train_loader, valid_loader, device, criterion, optimizer, epochs, step_interval):
    step = 0
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % step_interval == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss, validation_accuracy = validate(model, valid_loader, criterion, device)
                print(f"Epoch {epoch+1}/{epochs} - Training Loss: {running_loss/step_interval:.3f} - "
                      f"Validation Loss: {validation_loss/len(valid_loader):.3f} - "
                      f"Validation Accuracy: {validation_accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

# Model testing
def test_model(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct // total}%')

# Save the trained model
def save_model(model, save_directory, dataset):
    if save_directory is None:
        print("No save directory specified, skipping save.")
        return
    if not exists(save_directory):
        print("Directory does not exist, model not saved.")
        return
    model.class_to_idx = dataset.class_to_idx
    checkpoint = {
        'architecture': model.name,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_directory)

# Main function
def main():
    args = parse_arguments()

    # Data directories
    data_dir = 'flowers'
    train_dir, valid_dir, test_dir = [data_dir + '/' + x for x in ['train', 'valid', 'test']]

    # Preparing data
    train_dataset = training_data_transform(train_dir)
    valid_dataset = testing_data_transform(valid_dir)
    test_dataset = testing_data_transform(test_dir)

    train_loader = load_data(train_dataset)
    valid_loader = load_data(valid_dataset, False)
    test_loader = load_data(test_dataset, False)

    # Load model and setup
    model = load_model(args.architecture)
    model.classifier = setup_classifier(model, args.units)
    device = gpu_check(args.use_gpu)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    # Training
    train_network(model, train_loader, valid_loader, device, criterion, optimizer, args.training_epochs, 30)

    # Testing and saving
    test_model(model, test_loader, device)
    save_model(model, args.checkpoint_dir, train_dataset)

if __name__ == '__main__':
    main()
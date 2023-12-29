# Function to construct a deep learning model based on specified architecture and hidden units
def create_neural_network_model(architecture, num_hidden_units):
    
    import torch
    from torch import nn
    from torchvision import models
    from collections import OrderedDict

    # Define the number of output features for the classifier
    num_output_features = 102 

    # Load a pretrained model based on the specified architecture
    available_models = {
        "vgg19": models.vgg19,
        "resnet18": models.resnet18,
        "alexnet": models.alexnet,
        "vgg16": models.vgg16,
        "densenet161": models.densenet161,
        "inception_v3": models.inception_v3
    }
    model = available_models.get(architecture, models.vgg19)(pretrained=True)

    # Disable training for all the parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Determine the input size for the classifier
    classifier_input_size = next(model.classifier.parameters()).shape[1]

    # Define the classifier using an ordered dictionary
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, num_hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.4)),
        ('fc2', nn.Linear(num_hidden_units, num_output_features)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replace the model's classifier
    model.classifier = classifier

    return model
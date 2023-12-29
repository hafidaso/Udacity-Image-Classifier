import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import gpu_check
from torchvision import models

# Argument parsing function
def get_arguments():
    parser = argparse.ArgumentParser(description="Image Prediction")
    parser.add_argument('--input_image', type=str, help='Image path for prediction', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Model checkpoint path', required=True)
    parser.add_argument('--top_k_predictions', type=int, help='Top K predictions to return')
    parser.add_argument('--category_file', default='cat_to_name.json', help='Category to name mapping file')
    parser.add_argument('--use_gpu', default="gpu", help='Flag to use GPU')

    return parser.parse_args()

# Load model from checkpoint
def restore_model_from_checkpoint(filepath):
    model_data = torch.load(filepath)
    model_arch = model_data.get('architecture', 'vgg16')
    
    # Initialize pretrained model
    model = getattr(models, model_arch)(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Restore model data
    model.class_to_idx = model_data['class_to_idx']
    model.classifier = model_data['classifier']
    model.load_state_dict(model_data['state_dict'])

    return model

# Image preprocessing
def process_input_image(image_path):
    image = PIL.Image.open(image_path)
    original_width, original_height = image.size

    # Resize image
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        image = image.resize((256, int(256 / aspect_ratio)))
    else:
        image = image.resize((int(256 * aspect_ratio), 256))

    # Crop image
    crop_size = 244
    left = (image.width - crop_size) / 2
    top = (image.height - crop_size) / 2
    image = image.crop((left, top, left + crop_size, top + crop_size))

    # Normalize image
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - mean) / std
    return normalized_image.transpose(2, 0, 1)

# Predict the image
def predict_image(image, model, device, category_names, top_k):
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image_tensor)

    probabilities = torch.exp(output)
    top_probs, top_classes = probabilities.topk(top_k)

    # Convert indices to actual category names
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_classes.cpu().numpy()[0]]
    top_flower_names = [category_names[label] for label in top_labels]

    return top_probs.cpu().numpy()[0], top_labels, top_flower_names

# Print prediction results
def display_probabilities(flower_names, probabilities):
    for rank, (flower, prob) in enumerate(zip(flower_names, probabilities), 1):
        print(f"Rank {rank}: Flower: {flower}, Probability: {ceil(prob * 100)}%")

# Main function
def main():
    args = get_arguments()

    # Load category names
    with open(args.category_file, 'r') as f:
        category_to_name = json.load(f)

    # Load model and process image
    model = restore_model_from_checkpoint(args.model_checkpoint)
    processed_image = process_input_image(args.input_image)
    device = gpu_check(gpu_arg=args.use_gpu)

    # Prediction
    top_probs, top_labels, top_flowers = predict_image(processed_image, model, device, category_to_name, args.top_k_predictions)
    
    # Display results
    display_probabilities(top_flowers, top_probs)

if __name__ == '__main__':
    main()
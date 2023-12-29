import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)
    parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=True)
    parser.add_argument('--top_k',type=int,help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args


def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load("checkpoint.pth")
    
     model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    
    
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    img = PIL.Image.open(image)

    original_width, original_height = img.size
    
    if original_width < original_height:
        size=[256, 256**600]
    else: 
        size=[256**600, 256]
        
    img.thumbnail(size)
   
    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img)/255 

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img-mean)/std
        
    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img

ach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    #Converts two lists into a dictionary to print on screen
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))


def main():
    
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    image_tensor = process_image(args.image)
    
    device = check_gpu(gpu_arg=args.gpu);
    
    top_probs, top_labels, top_flowers = predict(image_tensor, model,device, cat_to_name,args.top_k)
    
    
    print_probability(top_flowers, top_probs)

if __name__ == '__main__': main()
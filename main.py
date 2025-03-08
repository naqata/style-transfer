import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image_transforms(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

def image_loader(image_name, device, transforms):
    image = Image.open(image_name)
    image = transforms(image).unsqueeze(0)
    return image.to(device, torch.float)

def image_show(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

def gram_matrix(tensor):
    a, b, c, d = tensor.size()
    tensor = tensor.view(a * b, c * d)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(a * b * c * d)

def get_features(model, image):
    layers = {'0': 'conv_1', '5': 'conv_2', '10': 'conv_3', '19': 'conv_4', '28': 'conv_5'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def main(content_path, style_path, output_dir, steps=1000, lr=0.02, content_weight=1, style_weight=1e6):
    device = get_device()
    transforms = get_image_transforms()
    
    content_image = image_loader(content_path, device, transforms)
    style_image = image_loader(style_path, device, transforms)
    
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    content_features = get_features(vgg, content_image)
    style_features = get_features(vgg, style_image)
    
    target_image = content_image.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([target_image], lr=lr)
    content_loss_fn = nn.MSELoss()
    style_loss_fn = nn.MSELoss()
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for step in range(steps):
        optimizer.zero_grad()
        target_features = get_features(vgg, target_image)
        content_loss = content_loss_fn(content_features['conv_1'], target_features['conv_1'])
        style_loss = sum(style_loss_fn(gram_matrix(target_features[layer]), gram_matrix(style_features[layer])) for layer in style_layers)
        
        total_loss = content_loss * content_weight + style_loss * style_weight
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            target_image.clamp_(0, 1)
        
        print(f'Epoch {step+1}: Loss = {total_loss.item():.4f}')
        
        
        output_path = os.path.join(output_dir, f'style_transfer_{step+1}.png')
        img = target_image[0].detach().cpu().permute(1, 2, 0).numpy()
        mpl.image.imsave(output_path, img)
        print(f'Saved: {output_path}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Style Transfer using PyTorch')
    parser.add_argument('content_path', type=str, help='Path to the content image')
    parser.add_argument('style_path', type=str, help='Path to the style image')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save output images')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--content_weight', type=float, default=1, help='Weight for content loss')
    parser.add_argument('--style_weight', type=float, default=1e5, help='Weight for style loss')
    
    args = parser.parse_args()
    main(args.content_path, args.style_path, args.output_dir, args.steps, args.lr, args.content_weight, args.style_weight)

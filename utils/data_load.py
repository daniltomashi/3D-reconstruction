import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import trimesh



# load and transform data
def load_image(image_path, mask_path, img_size=None):
    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')   # grayscale

    transform = T.Compose([T.Resize(img_size), T.ToTensor()]) if img_size else T.Compose([T.ToTensor()])
    img = transform(img)
    mask = transform(mask)

    return img, mask

# load 3d image
def load_3d_model(model_path):
    model = trimesh.load(model_path)
    return model

# visualize images
def visualize_data(img, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_title('Input Image')
    
    ax[1].imshow(mask.squeeze(0), cmap='gray')
    ax[1].set_title('Mask')

    plt.show()
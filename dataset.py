# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OccludedDataset(Dataset):
    def __init__(self, occluded_dir, original_dir, transform=None):
        self.occluded_dir = occluded_dir
        self.original_dir = original_dir
        self.transform = transform
        self.files = sorted(os.listdir(occluded_dir))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        occluded_name = self.files[idx]
        occluded_path = os.path.join(self.occluded_dir, occluded_name)

        if occluded_name.startswith("occluded_"):
            original_name = occluded_name.replace("occluded_", "", 1)
        else:
            original_name = occluded_name

        original_path = os.path.join(self.original_dir, original_name)

        # Check if original file exists
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original image not found: {original_path}")

        occluded_img = Image.open(occluded_path).convert('RGB')
        original_img = Image.open(original_path).convert('RGB')
        
        if self.transform:
            occluded_img = self.transform(occluded_img)
            original_img = self.transform(original_img)
        
        return occluded_img, original_img

def get_transforms(img_size=64):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

"""
pillow==8.3.2
timm==0.6.12
"""
import torch
import torchvision
from PIL import Image
import time

def load_img_tensor(img_path, resolution=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = Image.open(img_path).convert('RGB')   
    
    ## reproduce ILSVRC-2012 (center crop ratio = 0.875)
    transform= torchvision.transforms.Compose([
                    torchvision.transforms.Resize(int(resolution/0.875)),
                    torchvision.transforms.CenterCrop(resolution),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean, std)
                ])
    
    img_tensor = transform(img).unsqueeze(0)  # add batch channel
    return img_tensor

    
def load_caffee_labels(file_path):
    with open(file_path) as json_files:
        map = json.load(json_files) 
    return map

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
img_tensor = load_img_tensor("/root/mount-dir/dog.jpg")
print(DEVICE)
DEIT_MODELS = (
    'deit_tiny_patch16_224',
    'deit_tiny_distilled_patch16_224',
    'deit_small_distilled_patch16_224',
    'deit_base_patch16_384',
    'deit_base_distilled_patch16_384'
)
MODEL_NAME = DEIT_MODELS[1]
model = torch.hub.load('facebookresearch/deit:main', MODEL_NAME, pretrained=True)

"""
# Inference on GPU
img_tensor, model = img_tensor.to(DEVICE), model.to(DEVICE)
print(img_tensor.shape)
outputs = model(img_tensor)
print(torch.argmax(outputs[0]))
"""
# print(torch.max(outputs[0])) 
# probs = torch.nn.functional.softmax(outputs[0], dim=0)
# top_k=5
# prob, class_idx = torch.topk(probs, top_k)
# for i in range(top_k):
#     print("prob={} | class_idx={}".format(prob[i].item(), class_idx))
# print(torch.argmax(pred, dim=0))

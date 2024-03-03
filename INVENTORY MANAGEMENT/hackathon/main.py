import cv2
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image

model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")

labels = list(model.config.id2label.values())

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "path/image.jpeg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(image) 
input_tensor = preprocess(pil_image).unsqueeze(0)

outputs = model(input_tensor)

predicted_idx = torch.argmax(outputs.logits, dim=1).item()

predicted_label = labels[predicted_idx]

print("Detected label:", predicted_label)
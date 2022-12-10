import torchvision
from torchvision import transforms
img_demo_path = '../data/faces/1.jpg'
img = torchvision.io.read_image(img_demo_path)

compose = [
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
]
transform = transforms.Compose(compose)

img = transform(img)
print(img)
print(img.shape)
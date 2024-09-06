import torch
import torchvision
from PIL import Image

image_path = "imgs/dog.png"
image = Image.open(image_path)

dict_sample = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
               8: 'ship', 9: 'truck'}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Resize the image to 224x224
image = image.resize((224, 224))

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

model = torch.load("network_9")
print(model)
image = torch.unsqueeze(image, dim=0).to(device)
print(image.shape)

# Set the model to evaluation mode and disable gradient computation
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1).item())
print(dict_sample.get(output.argmax(1).item()))

from jetcam.csi_camera import CSICamera
import torchvision.transforms as transforms
from dataset import ImageClassificationDataset
import torch
import torchvision
from utils import preprocess
import torch.nn.functional as F
import PIL.Image
import numpy as np

BATCH_SIZE = 8
LEARNING_RATE = 0.1
MOMENTUM = 0.9
EPOCHS = 1

CATEGORIES = ['Banana', 'Blueberry', 'Raspberry', 'Watermelon', 'Pineapple']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageClassificationDataset('./data/fruits', CATEGORIES, TRANSFORMS)

device = torch.device('cuda')

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def live(model, camera):
    global dataset
	camera.running = True
    while True:
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        prediction = dataset.categories[category_index]
        for i, score in enumerate(list(output)):
            pass


def train(is_training, model):
	global BATCH_SIZE, LEARNING_RATE, MOMENTUM, EPOCHS, dataset
    
    try:
		epoch = EPOCHS
		optimizer = torch.optim.Adam(model.parameters())

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        if is_training:
            model = model.train()
        else:
            model = model.eval()
        while epoch > 0:
            i = 0
            sum_loss = 0.0
            error_count = 0.0
            for images, labels in iter(train_loader):
                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                if is_training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images)

                # compute loss
                loss = F.cross_entropy(outputs, labels)

                if is_training:
                    # run backpropogation to accumulate gradients
                    loss.backward()

                    # step optimizer to adjust parameters
                    optimizer.step()

                # increment progress
                error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
                count = len(labels.flatten())
                i += count
                sum_loss += float(loss)
				print(str(i / len(dataset)), "loss:", str(sum_loss / i), "acc:", str(1.0 - error_count / i))
                
            if is_training:
                epoch -= 1
            else:
                break
    except e:
        pass
    model = model.eval()


if __name__ == '__main__':
	camera = CSICamera(width=224, height=224, capture_device=0)
	print("camera configured")

	model = torchvision.models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(512, len(dataset.categories))
	model = model.to(device)
	print("model configured")


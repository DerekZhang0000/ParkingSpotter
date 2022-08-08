from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

customTransforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.ColorJitter(saturation=0.2),
    transforms.Resize(900),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

Train_dataset = core.Dataset("trainingImages/", transform=customTransforms)
Test_dataset = core.Dataset("testImages/")
loader = core.DataLoader(Train_dataset, batch_size=1, shuffle=True)


def train(e=25, lr=0.001, loss=False, prediction=False, forceSave=False):
    try:
        model = core.Model.load("ParkingSpotterModel.pt", ["Occupied", "Vacant"])
    except:
        model = core.Model(["Occupied", "Vacant"])

    losses = model.fit(loader, Test_dataset, epochs=e, lr_step_size=5, learning_rate=lr, verbose=True)

    if loss:
        plt.plot(losses)
        plt.show()

    if prediction:
        predictions()

    if forceSave:
        model.save("ParkingSpotterModel.pt")
    else:
        userInput = input("Save model? (y/n) ")
        while userInput not in ["y", "n"]:
            userInput = input("Save model? (y/n) ")
        if userInput == "y":
            model.save("ParkingSpotterModel.pt")

def predictions(n=10):
    model = core.Model.load("ParkingSpotterModel.pt", ["Occupied", "Vacant"])
    for image in [image for image in os.listdir("testImages/") if image.endswith(".PNG")][:n]:
        image = utils.read_image("testImages/" + image)
        predictions = model.predict(image)
        labels, boxes, scores = predictions
        thresh = .7
        filteredIndices = np.where(scores>thresh)
        filteredBoxes = boxes[filteredIndices]
        numList = filteredIndices[0].tolist()
        filteredLabels = [labels[i] + " " + str(scores[i].data)[7:-1] for i in numList]
        show_labeled_image(image, filteredBoxes, filteredLabels)
        plt.show()

train(25, 0.005, True, True, True)
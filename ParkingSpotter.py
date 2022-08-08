from detecto import core, utils
from detecto.visualize import show_labeled_image
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import os

def makePrediction(imagePath):
    print("Loadling model...")
    model = core.Model.load("ParkingSpotterModel.pt", ["Occupied", "Vacant"])
    thresh = -1
    while thresh > 1 or thresh < 0:
        thresh = float(input("Enter threshold value: "))
    try:
        image = utils.read_image(imagePath)
    except:
        print("Error: Could not open image")
        return
    predictions = model.predict(image)
    labels, boxes, scores = predictions
    filteredIndices = np.where(scores > thresh)
    filteredBoxes = boxes[filteredIndices]
    numList = filteredIndices[0].tolist()
    filteredLabels = [labels[i] for i in numList]
    spaces = len(filteredLabels)
    vacant = len([space for space in filteredLabels if space == "Vacant"])
    occupied = len([space for space in filteredLabels if space == "Occupied"])
    print("There are {} spaces in the image.".format(spaces))
    print("{} are vacant and {} are occupied.".format(vacant, occupied))
    show_labeled_image(image, filteredBoxes, filteredLabels)
    plt.show()
    print("Exited...")

print("ParkingSpotter v1.0")
makePrediction(filedialog.askopenfilename(initialdir=os.getcwd()))
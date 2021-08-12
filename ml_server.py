
# creating a basic flask server

import json
import torch
import torchvision
import random
import numpy as np
from ChestXRayDataset import ChestXRayDataset

from flask import Flask, request

app = Flask(__name__)

# create model
resnet18model = torchvision.models.resnet18(pretrained=False)

# change the number of output features for last layer  = 3
resnet18model.fc = torch.nn.Linear(in_features=512, out_features=3)

# load the model
resnet18model.load_state_dict(torch.load('covidRadiography.pth'))

dirs = {
    'normal': 'COVID-19 Radiography Database/normal',
    'viral': 'COVID-19 Radiography Database/viral',
    'covid': 'COVID-19 Radiography Database/covid'
}

transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)), # resize the images to use them in pre-trained model
    torchvision.transforms.ToTensor(), # convert the image to tensor
    # normalize the data
    # using the mean, std same as resnet-18 model to take advantage of pretrained weights
    torchvision.transforms.Normalize(mean = [0.485,0.456,0.406],
                                     std = [0.229,0.224,0.225])
])

batch_size = 6
dataset = ChestXRayDataset(dirs,transformation)
dl = torch.utils.data.DataLoader(dataset,
                                 batch_size= batch_size,
                                 shuffle=True)

def get_prediction():
    resnet18model.eval()
    images, labels = next(iter(dl))
    outputs = resnet18model(images)
    _, preds = torch.max(outputs,1)
    return images.numpy().tolist(),labels.numpy().tolist(),preds.numpy().tolist()


@app.route('/', methods = ['GET','POST'])

def index():
    if request.method == 'POST':
        images,labels,preds = get_prediction()
        
        return json.dumps({
            'predictions': preds,
            'images': images,
            'labels': labels
        })
    return "Welcome to ML server"

if __name__ == '__main__':
    app.run()

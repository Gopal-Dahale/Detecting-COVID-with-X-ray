
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ChestXRayDataset import ChestXRayDataset

from flask import Flask, request

app = Flask(__name__)

class_names = ['covid','normal','viral']
batch_size = 6

# create model
resnet18model = torchvision.models.resnet18(pretrained=False)

# change the number of output features for last layer  = 3
resnet18model.fc = torch.nn.Linear(in_features=512, out_features=3)

# load the model
resnet18model.load_state_dict(torch.load('covidRadiography.pth'))

dirs = {
    'normal': 'COVID-19 Radiography Database/test/normal',
    'viral': 'COVID-19 Radiography Database/test/viral',
    'covid': 'COVID-19 Radiography Database/test/covid'
}

transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)), # resize the images to use them in pre-trained model
    torchvision.transforms.ToTensor(), # convert the image to tensor
    # normalize the data
    # using the mean, std same as resnet-18 model to take advantage of pretrained weights
    torchvision.transforms.Normalize(mean = [0.485,0.456,0.406],
                                     std = [0.229,0.224,0.225])
])

dataset = ChestXRayDataset(dirs,transformation)
dl = torch.utils.data.DataLoader(dataset,batch_size= batch_size,shuffle=True)

def get_prediction():
    resnet18model.eval()
    images, labels = next(iter(dl))
    outputs = resnet18model(images)
    _, preds = torch.max(outputs,1)
    return images.numpy().tolist(),labels.numpy().tolist(),preds.numpy().tolist()

def predict():
    images,labels,preds = get_prediction()
    return {
        'predictions': preds,
        'images': images,
        'labels': labels
    }

def main():
    st.title("Detecting COVID-19 with Chest X-Ray using PyTorch")
    st.write("The label on the x-axis is the ground truth")
    st.write("The label on the y-axis is the prediction. Green color means that the prediction matches the ground truth otherwise the color is red.")

    if st.button('Get random predictions'):
        response = predict()
        preds = response.get('predictions')
        images = response.get('images')
        labels = response.get('labels')
        plt.figure(figsize=(12,8))

        # display a batch
        for i,image in enumerate(images):
            plt.subplot(1,batch_size,i+1,xticks = [],yticks = [])

            # convert tensor image to numpy array
            # the format for resnet-18 model is channel first
            image = np.array(image).transpose(1,2,0)

            # undo the normalize
            mean = np.array([0.485,0.456,0.406])
            std = np.array([0.229,0.224,0.225])
            image = image*std + mean
            image = np.clip(image,0,1)
            plt.imshow(image)

            # use x label to should ground truth
            # use y label to show prediction
            # green prediction - correct
            # red prediction - incorrect
            color = ""
            if preds[i] == labels[i]:
                color = 'green'
            else:
                color = 'red'

            plt.xlabel(str(class_names[int(labels[i])]))
            plt.ylabel(str(class_names[int(preds[i])]), color = color)
        st.pyplot()

if __name__=='__main__': 
    main()
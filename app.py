
# import streamlit as st
# from PIL import Image
# st.set_option('deprecation.showfileUploaderEncoding', False)

# st.title("Upload + Classification Example")
# file_up = st.file_uploader("Upload an image", type="png")

# if file_up is not None:
#     image = Image.open(file_up)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("addy")

import streamlit as st
import json
import random
import requests
import matplotlib.pyplot as plt
import numpy as np

URI = 'http://127.0.0.1:5000/'
class_names = ['covid','normal','viral']
batch_size = 6

st.title("Detecting COVID-19 with Chest X-Ray using PyTorch")
st.write("The label on the x-axis is the ground truth")
st.write("The label on the y-axies is the prediction. Green color means that the prediction matches the ground truth otherwise the color is red.")

if st.button('Get random predictions'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
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

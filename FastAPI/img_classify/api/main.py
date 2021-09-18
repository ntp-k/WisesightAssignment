from fastapi import FastAPI
from fastapi import UploadFile, File
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import tensorflow
import keras
from keras.models import model_from_json
import aiofiles

app = FastAPI()


def loadModel():
    global loaded_model
    
    #load model
    json_file = open('model_VGG16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights
    loaded_model.load_weights("model_VGG16.h5")


loadModel()


def preprocess(filename):
    IMAGE_WIDTH=128
    IMAGE_HEIGHT=128
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    predict_df = pd.DataFrame([filename], columns=['file'])
    predict_datagen = ImageDataGenerator(rescale=1./255)

    predict_gen = predict_datagen.flow_from_dataframe(
        predict_df,
        x_col='file',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=1)
    
    return predict_gen

def predict(filename, model):
    predict_gen = preprocess(filename)
    result = model.predict(predict_gen, steps=1)

    if result[0][0] > 0.5:
        return 'this is a cat'
    elif result[0][1] > 0.5:
        return 'this is a dog'
    else:
        return 'others'


@app.post('/api/classify')
async def predict_img(file: UploadFile = File(...)):
    async with aiofiles.open(file.filename, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    result = predict(file.filename, loaded_model)

    return { 'success' : 'true', 'class' : result }
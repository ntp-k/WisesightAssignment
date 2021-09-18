from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pythainlp import word_tokenize
from pythainlp.ulmfit import process_thai
#docker run -d -p 80:80 test_image_wisesight

app = FastAPI()


def loadResource():
    global model_loaded
    global vectorizer_loadded
    global scaler_loadded
    
    model_loaded = joblib.load('model_LogR.joblib')
    vectorizer_loadded = joblib.load('vectorizer.joblib')
    scaler_loadded = joblib.load('scaler.joblib')


loadResource()


def preprocess(text):
    text_processed = process_thai(text)
    text_wc = len(text_processed)
    text_uwc = len(set(text_processed))
    
    data = {'text': [text], 'wc': [text_wc], 'uwc': [text_uwc]}
    df_predict = pd.DataFrame(data)
    
    text_predict = vectorizer_loadded.transform(df_predict['text'])
    num_predict = scaler_loadded.transform(df_predict[["wc","uwc"]].astype(float))
    x_predict = np.concatenate([num_predict,text_predict.toarray()],axis=1)

    return x_predict


def predict(text, model):
    text_predict = preprocess(text)
    result = model_loaded.predict(text_predict)
    
    return result[0]


@app.get('/api/analyze')
def get_root(text: str):
    result = predict(text, model_loaded)

    return { 'success' : 'true', 'sentiment' : result }
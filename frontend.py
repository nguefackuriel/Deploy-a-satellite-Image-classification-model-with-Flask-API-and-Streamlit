import streamlit as st
from PIL import Image
import requests


# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry


# session = requests.Session()
# retry = Retry(connect=3, backoff_factor=0.5)

st.title("Satellite Image Classification")

upload = st.file_uploader("Upload an image",
                           type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)
#st.write(upload.getvalue())
if upload:
    files = {"file" :  upload.getvalue()}

    req = requests.post("http://127.0.0.1:8080/predict", files=files)
    #st.write(req)

    result = req.json()
    rec = result["prediction"]
    #rec = req
    # prob_recyclable = rec * 100      
    # prob_organic = (1-rec)*100

    c1.image(Image.open(upload).convert('RGB'))
    
    c2.write(f"The image you uploaded is {rec} ")

    

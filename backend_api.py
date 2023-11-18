from flask import Flask, jsonify, request
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



app = Flask(__name__)


@app.route("/", methods=["GET"])


@app.before_first_request
def load():
    model_path = "resnext101.pth"
    model = torch.load(model_path,map_location=torch.device('cpu'))
    return model

# Load model
model = load()

def preprocess(img):
    transforms_image = transforms.Compose([ 
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = transforms_image(img)
    img = img.unsqueeze(0).to(device)
    
    return img




classes = ['cloudy', 'desert', 'green_area', 'water']

@app.route("/predict", methods=['POST'])
def predict():
    # Take the image
    file = request.files['file']
    image = file.read()

    # Open the image
    img = Image.open(io.BytesIO(image))#Image.open(image)#
    img = img.convert('RGB')
    #preprocess the image
    img_processed = preprocess(img)

    # prediction
    yb = model(img_processed)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    

    rec = classes[preds[0].item()]

    return jsonify({"prediction" : rec})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
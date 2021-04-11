import io
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, jsonify, request, render_template
from PIL import Image

app = Flask(__name__)

# Modelling Task
model = models.vgg16(pretrained=True)

# Change to model state pict path on your computer
model_state_dict_path = r"/Users/kylelee/Desktop/Patho-ML_App/vgg_subclass_model_state_dict_02252021.pt"
model.load_state_dict(torch.load(model_state_dict_path,map_location=torch.device('cpu')))
model.eval()

class_names = ["Adenosis","Ductal Carcinoma", "Fibroadenoma", "Lobular Carcinoma","Mucinous Carcinoma","Papillary Carcinoma","Phyllodes Tumor","Tubular Adenoma"]

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

imsize = 256
loader = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def transform_image(image_bytes):
	my_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	#tensor = image_loader(loader, image_bytes)
	outputs = model.forward(tensor)
	_, prediction = torch.max(outputs, 1)
	return class_names[prediction]

diseases = {
	"Adenosis": "Benign tissue",
	"Ductal Carcinoma": "Malignant tissue",
	"Fibroadenoma": "Benign tissue",
	"Mucinous Carcinoma": "Malignant tissue",
	"Papillary Carcinoma": "Malignant tissue",
	"Phyllodes Tumor": "Benign tissue",
	"Tubular Adenoma": "Benign tissue",
}

# Treat the web process
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files.get('file')
		if not file:
			return
		img_bytes = file.read()
		prediction_name = get_prediction(img_bytes)
		return render_template('result.html', name=prediction_name.lower(), description=diseases[prediction_name])

	return render_template('index.html')


if __name__ == '__main__':
	app.run(debug=True)
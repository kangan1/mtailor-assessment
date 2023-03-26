import onnxruntime
from torchvision import transforms
from torch import Tensor
import onnx

import numpy as np
from PIL import Image
import os

import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper

# model_dir ="./mnist"
model="model.onnx"
# path=sys.argv[1]
# path = ".\\images\\n01667114_mud_turtle.JPEG"

path = ".\\images\\n01440764_tench.jpeg"
 
#Preprocess the image
img = cv2.imread(path)
img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
img.resize((1, 3, 224, 224))


# image = image.resize((224, 224), resample=Image.BILINEAR)
# image = np.array(image).astype(np.float32) / 255.0
# image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
# image = image.transpose((2, 0, 1))
# image = image.astype(np.float32)



data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
#print(input_name)
#print(output_name)
 
result = session.run([output_name], {input_name: data})
prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
print(prediction)
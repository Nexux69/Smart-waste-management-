cam_port = None
from cv2 import *
cam_port = 0
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
result = None
def getPredictionsFromModel():
    model_path = "/home/pi/Desktop/Grok-Downloads/wastemanagement.tflite"
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    categories = ["Biodegradable", "NonBiodegradable", "No Object Found"]
    input_details = interpreter.get_input_details()
    dtype =  input_details[0]['dtype']
    Shape = input_details[0]['shape']
    frame = cv2.imread("/home/pi/Desktop/Grok-Downloads/image.jpg")
    image = cv2.resize(frame, (224, 224))
    image = image.reshape(Shape)
    image = image.astype(dtype)
    image = image / 255.0
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_tensor_index, image)
    interpreter.invoke()
    output_tensor_index = interpreter.get_output_details()[0]['index']
    prediction = interpreter.get_tensor(output_tensor_index)[0]
    class_idx = np.argmax(prediction)
    class_label = categories[class_idx]
    confidence = prediction[class_idx] * 100
    return class_label, confidence
def getResults():
    class_label, confidence = getPredictionsFromModel()
    print(f"Class: {class_label}, Confidence: {confidence:.2f}%")
    return class_label, confidence
while True:
  cam_port = 0
  cam = VideoCapture(cam_port)
  result, image = cam.read()
  if result:
  	imwrite("/home/pi/Desktop/Grok-Downloads/image.jpg", image)
  cam.release()
  getPredictionsFromModel()
  print('Result:' + str(getResults()))
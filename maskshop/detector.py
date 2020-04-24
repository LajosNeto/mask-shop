"""
Image detectors.
Includes a detector for face box detection and facial expression detector.
"""

# Author:
# Lajos Neto <lajosneto@gmail.com>


import numpy as np
import cv2

from tensorflow import keras


LABELS = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}


class DNNFaceDetector():
    def __init__(self):
        model_file = 'model_output/opencv_face_detector_uint8.pb'
        config_file = 'model_output/opencv_face_detector.pbtxt'
        self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    
    def run(self, image):
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        # detect face position within the image frame
        detections = self.net.forward()
        (h, w) = image.shape[:2]
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # extract the detected face (only face portion)
        face_image = image[startY:endY, startX:endX]
        # draw rectangle around detected face (original image)
        box_image = cv2.rectangle(image,(startX,startY),(endX,endY),(255,0,0),2)
        return (box_image, face_image, (startX, startY, endX, endY))

class ExpressionDetector():
    def __init__(self, model):
        self.model = model
    
    def pre_process(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, dsize=(48,48), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image,[0,3])
        image = (image - np.mean(image))/np.std(image)
        return image
    
    def run(self, image):
        image = self.pre_process(image)
        predictions = self.model.predict(image)
        predicted_label = np.argmax(predictions)
        return (predicted_label, predictions)
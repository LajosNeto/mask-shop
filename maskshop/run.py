"""
Main camera routine
"""

# Author:
# Lajos Neto <lajosneto@gmail.com>


import argparse

import cv2
import numpy as np

from detector import DNNFaceDetector, ExpressionDetector
from utils.plot_utils import generate_probs_plot, apply_probs_overlay
from models import list_models, load_model_h5


def gen_frame(camera, face_mode, expression_model):
    while(True):
        ret,frame = camera.read()
        box, face = face_mode.run(frame)
        predicted_label, predictions = expression_model.run(face)
        apply_probs_overlay(box, predictions[0])
        return box


if __name__ == '__main__':
    model_generator_argparse = argparse.ArgumentParser()
    model_generator_argparse.add_argument('--model', type=str, required=True)
    args = model_generator_argparse.parse_args()
    model_name = args.model

    model = load_model_h5(model_name)
    if(model):
        camera = cv2.VideoCapture(0)
        face_detector = DNNFaceDetector()
        expression_detector = ExpressionDetector(model)
        while(True):
            cv2.imshow('frame2',gen_frame(camera, face_detector, expression_detector))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
    else:
        print(f"Invalid model, please, provide one of the following options : {', '.join(list_models())}")
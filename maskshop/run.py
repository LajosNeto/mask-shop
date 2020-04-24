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
from masker import Masker


def gen_frame(camera, face_model, expression_model, masker, debug_mode):
    while(True):
        ret,frame = camera.read()

        box, face, position = face_model.run(frame.copy())
        predicted_label, predictions = expression_model.run(face)

        if debug_mode:
            apply_probs_overlay(box, predictions[0])
            return box
        else:
            apply_probs_overlay(frame, predictions[0])
            if(predictions[0][predicted_label] > 0.7):
                frame = masker.apply_mask(frame, position, predicted_label)
        return frame


if __name__ == '__main__':
    model_generator_argparse = argparse.ArgumentParser()
    model_generator_argparse.add_argument('--model', type=str, required=True)
    model_generator_argparse.add_argument('-d', '--debug', action='store_true')
    args = model_generator_argparse.parse_args()
    model_name = args.model
    debug_mode = args.debug

    model = load_model_h5(model_name)
    if(model):
        camera = cv2.VideoCapture(0)
        face_detector = DNNFaceDetector()
        expression_detector = ExpressionDetector(model)
        masker = Masker()
        while(True):
            cv2.imshow('frame2',gen_frame(camera, face_detector, expression_detector, masker, debug_mode))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
    else:
        print(f"Invalid model, please, provide one of the following options : {', '.join(list_models())}")
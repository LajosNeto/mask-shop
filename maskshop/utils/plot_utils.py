"""
Plot and text formatters utils.
"""

# Author:
# Lajos Neto <lajosneto@gmail.com>


import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_COLOR = (204, 204, 204)
LINE_TYPE = 2
VERTICAL_MARGIN = 30
START_X = 10
START_Y = 500


def proba_to_percentage(prob):
    return str(round(prob*100,5)) + " %"

def generate_probs_plot(probs):
    return [
        f"angry : {proba_to_percentage(probs[0])}",
        f"disgust : {proba_to_percentage(probs[1])}",
        f"fear : {proba_to_percentage(probs[2])}",
        f"happy : {proba_to_percentage(probs[3])}",
        f"sad : {proba_to_percentage(probs[4])}",
        f"surprise : {proba_to_percentage(probs[5])}",
        f"neutral : {proba_to_percentage(probs[6])}"]

def apply_probs_overlay(frame, probs):
    for i, prob_line in enumerate(generate_probs_plot(probs)):
        y = START_Y + (i*VERTICAL_MARGIN)
        cv2.putText(frame, prob_line, (START_X, y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
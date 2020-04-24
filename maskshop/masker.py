"""
Mask image applier class
"""

# Author:
# Lajos Neto <lajosneto@gmail.com>


import cv2


MASKS = ['angry.png', 'disgust.png', 'fear.png', 'happy.png', 'sad.png', 'surprise.png', 'neutral.png']


class Masker():

    def __init__(self):
        self.masks = [cv2.imread('masks/'+mask, -1) for mask in MASKS]
    
    def apply_mask(self, frame, position, emotion):
        mask = self.masks[emotion]
        alpha_s = mask[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        middle = int((position[2] - position[0])/2) + position[0]

        try:
            for c in range(0, 3):
                frame[(position[3]-300):(position[3]), (middle-150):(middle+150), c] = (alpha_s * mask[:, :, c] + alpha_l * frame[(position[3]-300):(position[3]), (middle-150):(middle+150), c])
        except:
            pass

        return frame
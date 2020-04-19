import cv2
import numpy as np

from detector import DNNFaceDetector, ExpressionDetector

from tensorflow import keras


def gen_frame(camera, face_mode, expression_model):
    while(True):
        ret,frame = camera.read()
        box, face = face_mode.run(frame)
        predicted_label, predictions = expression_model.run(face)
        print(predicted_label)
        return box


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    
    face_detector = DNNFaceDetector()
    expression_detector = ExpressionDetector()

    while(True):
        cv2.imshow('frame2',gen_frame(camera, face_detector, expression_detector))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
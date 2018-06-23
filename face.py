from detect import ObjectDetector

import dlib
import cv2
FACE_PAD = 5

class FaceDetectorDlib(ObjectDetector):
    def __init__(self, model_name):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_name)

    def run(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 2)
        images = []
        bb = []
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            bb.append((x,y,w,h))
            images.append(self.sub_image(img, x, y, w, h))

        print('%d faces detected' % len(images))
        return images, bb

    def sub_image(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        return roi_color
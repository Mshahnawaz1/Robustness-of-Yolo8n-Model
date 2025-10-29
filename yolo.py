from ultralytics import YOLO 
import cv2

from utils import *

YOLO_MODEL_NAME = 'models/yolov8n.pt'
IMG = "image/4.jpg"        # Using the fast Nano model as an example

model = YOLO(YOLO_MODEL_NAME)
print(f"Model loaded: {YOLO_MODEL_NAME}")

SIGMA = [0, 10, 20, 30, 40, 50]

class YoloFaceDetector:
    def __init__(self, model_path=YOLO_MODEL_NAME, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, image):
        results = self.model.predict(image, save=False, verbose=False, conf=self.conf_threshold)
    
        return results
    def add_noise_annotate(self, image, sigma: list):
        noisy_images = [add_gaussian_noise(image, sigma=s) for s in sigma]
        annotated_images = [draw_yolo_boxes(self.detect_faces(img), img) for img in noisy_images]
        return annotated_images

    def show(self, image_path):
        img = read_image(image_path)
        if img is None:
            print(f"Error: Unable to load image at {image_path}")
            return
        
        results = self.detect_faces(img)
        annotated = draw_yolo_boxes(results, img)
        cv2.imshow("YOLO Face Detection", annotated)
        cv2.waitKey(0)
    

if __name__ == "__main__":
    img_path = "image/4.jpg"
    yolo = YoloFaceDetector()
    yolo.show(img_path)
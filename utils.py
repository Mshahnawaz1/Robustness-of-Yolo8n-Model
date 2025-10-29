import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt


def draw_yolo_boxes(results: list, img: np.ndarray) -> np.ndarray:
    """
    Draws bounding boxes and confidence scores on the image based on YOLO prediction results.

    Args:
        results (list): The list of result objects returned by model.predict().
        img (np.ndarray): The original image (BGR format).

    Returns:
        np.ndarray: The image with boxes and labels drawn, or the original image if no faces are detected.
    """
    if results and len(results[0].boxes) > 0:
        annotated_img = results[0].plot()
        
        # Optional: Print diagnostic info (can be moved outside if preferred)
        boxes = results[0].boxes
        num_detections = len(boxes)
        min_conf = boxes.conf.min().item()
        
        print(f"Total Faces Detected: {num_detections}")
        print(f"Lowest Confidence Score: {min_conf:.4f}")
        
        return annotated_img
    
    else:
        return img # Return original image if no detections
    

def read_image(image_path):
    """Read an image from a file."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, (360, 360))
    return img

def add_gaussian_noise(image, mean=0, sigma=0.1):
    # Normalize image to [0,1]
    img = image / 255.0  
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img + gauss
    # Clip values to [0,1] range then scale back
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    return noisy

def show_images(title, images):
    """Display a list of images with a title."""
    plt.figure(figsize=(12, 6))
    plt.suptitle(title)
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    plt.show()


def add_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies Gaussian blur to the image.
    Kernel size must be a positive odd integer. If 0, original image is returned.
    """
    if kernel_size <= 1:
        return image.copy()
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image
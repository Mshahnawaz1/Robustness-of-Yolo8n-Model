import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_yolo_boxes(results: list, img: np.ndarray) -> np.ndarray:
    """
    Draws bounding boxes and confidence scores on the image based on YOLO prediction results.

    Args:
        results (list): The list of result objects returned by model.predict().
        img (np.ndarray): The original image (BGR format).

    Returns:
        np.ndarray: The image with boxes and labels drawn, or the original image if no detections.
    """
    if results and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        annotated_img = results[0].plot()

        # Optional diagnostic info
        boxes = results[0].boxes
        num_detections = len(boxes)
        min_conf = boxes.conf.min().item() if hasattr(boxes, 'conf') else None

        print(f"Detections: {num_detections}, Min confidence: {min_conf:.2f}" if min_conf else f"Detections: {num_detections}")

        return annotated_img
    else:
        return img  # Return original image if no detections

    

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

def add_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies Gaussian blur to the image.
    Kernel size must be a positive odd integer. If 0, original image is returned.
    """
    if kernel_size <= 1:
        return image.copy()
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        # If it's even, make it the next odd number (e.g., 4 becomes 5)
        kernel_size += 1
    
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image


def add_jpeg_compression(image: np.ndarray, quality_factor: int) -> np.ndarray:
    """
    Applies JPEG compression degradation to the image using a quality factor.
    Quality factor must be an integer between 0 (worst quality) and 100 (best quality).
    If quality_factor is 100, the original image is essentially returned (lossless).
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]

    is_success, encoded_image = cv2.imencode('.jpg', image, encode_param)

    if not is_success:
        print("Warning: JPEG encoding failed. Returning original image.")
        return image.copy()
    
    compressed_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if compressed_image is None:
        print("Warning: JPEG decoding failed. Returning original image.")
        return image.copy()
    
    return compressed_image


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size

    # Salt noise (white pixels)
    num_salt = int(total_pixels * salt_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper noise (black pixels)
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy

def add_poisson_noise(image, intensity=1.0):
    """
    Adds controllable Poisson noise (shot noise) to an image.

    Args:
        image (np.ndarray): Input image (uint8, range 0–255).
        intensity (float): Exposure level controlling noise strength.
                           Lower intensity => stronger noise.
                           Must be > 0.

    Returns:
        np.ndarray: Noisy image (uint8, same shape as input).
    """
    if intensity <= 0:
        raise ValueError("Intensity must be greater than 0.")

    # Normalize to [0,1]
    img = image.astype(np.float32) / 255.0

    # Simulate exposure: fewer photons at low intensity → more noise
    scaled = img * intensity
    noisy = np.random.poisson(scaled * 255.0) / (255.0 * intensity)

    # Clip and rescale back to 0–255
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)

def _show(values, images, title):
    """
    Metadata:
    sigma: list of sigma values used for Gaussian noise
    quality_factor: list of quality factors used for JPEG compression

    Values:list => sigma, quality_factor, blur_kernel_size
    type : str => gaussian_noise, jpeg_compression, motion_blur
    Robustness Metrics:
"""
    plt.figure(figsize=(18, 12))
    for i, (s, img) in enumerate(zip(values, images)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if i == 0:
            plt.title(f'{title} Truth')
        else:
            plt.title(f'Noise: {s}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
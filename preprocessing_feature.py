import os
import torch
import cv2
import numpy as np
from PIL import Image

base_folder = "C:\\Users\\khand\\Places365_Dataset"
splits = {
    "train": os.path.join(base_folder, "train_100"),
    "val": os.path.join(base_folder, "val_100"),
    "test": os.path.join(base_folder, "test_100")
}

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def resize_and_normalize(image, size=(256, 256)):
    """Resize and normalize an image to the specified size and [0, 1] range."""
    img = image.resize(size, Image.LANCZOS)
    img_array = np.asarray(img) / 255.0 
    return Image.fromarray((img_array * 255).astype('uint8'))

def apply_yolo_mask(image_path):
    """Detect objects using YOLO and mask detected regions in the image."""
    results = model(image_path)
    img = cv2.imread(image_path)

    for bbox in results.xyxy[0]:  
        x1, y1, x2, y2 = map(int, bbox[:4].tolist())
        img[y1:y2, x1:x2] = 0 
    return img

def apply_canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges

def apply_contour_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img

def process_images(input_folder, output_folder_base):
    os.makedirs(output_folder_base, exist_ok=True)
    
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        img = Image.open(img_path).convert("RGB")
        resized_normalized_img = resize_and_normalize(img)
        resized_normalized_path = os.path.join(output_folder_base, "resized_normalized")
        os.makedirs(resized_normalized_path, exist_ok=True)
        resized_normalized_img.save(os.path.join(resized_normalized_path, img_name))
        
        yolo_masked_img = apply_yolo_mask(img_path)
        yolo_masked_path = os.path.join(output_folder_base, "yolo_masked")
        os.makedirs(yolo_masked_path, exist_ok=True)
        cv2.imwrite(os.path.join(yolo_masked_path, img_name), yolo_masked_img)
        
        edge_detected_img = apply_canny_edge_detection(yolo_masked_img)
        edge_detected_path = os.path.join(output_folder_base, "edge_detected")
        os.makedirs(edge_detected_path, exist_ok=True)
        cv2.imwrite(os.path.join(edge_detected_path, img_name), edge_detected_img)
        
        contour_detected_img = apply_contour_detection(yolo_masked_img)
        contour_detected_path = os.path.join(output_folder_base, "contour_detected")
        os.makedirs(contour_detected_path, exist_ok=True)
        cv2.imwrite(os.path.join(contour_detected_path, img_name), contour_detected_img)

    print(f"Processing complete for {input_folder}. Outputs saved to {output_folder_base}")

for split_name, input_folder in splits.items():
    output_folder_base = os.path.join(base_folder, f"{split_name}_processed")
    print(f"Starting processing for {split_name} data...")
    process_images(input_folder, output_folder_base)
    print(f"Finished processing for {split_name} data.\n")

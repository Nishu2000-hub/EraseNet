
import os
import cv2
from ultralytics import YOLO
import numpy as np

TRAIN_IMAGE_LIST = "train_image_paths.txt"
TEST_IMAGE_LIST = "test_image_paths.txt"
MASK_DIR = "masks"
os.makedirs(MASK_DIR, exist_ok=True)

def generate_masks(image_list_file):
    model = YOLO('yolov8n.pt')
    model.to('cpu')
    mask_count = 0

    with open(image_list_file, 'r') as f:
        image_paths = [line.strip() for line in f]

    for img_path in image_paths:
        results = model(img_path, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=-1)

        base_name = os.path.basename(img_path)
        mask_name = os.path.splitext(base_name)[0] + "_mask.png"
        cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask)
        mask_count += 1

    print(f"Masks generated: {mask_count} for {image_list_file}.")

if __name__ == "__main__":
    print("Generating masks for training images...")
    generate_masks(TRAIN_IMAGE_LIST)
    print("Generating masks for testing images...")
    generate_masks(TEST_IMAGE_LIST)



# import os
# import cv2
# import torch
# from ultralytics import YOLO
# import numpy as np

# IMAGE_LIST = "train_image_paths.txt" 
# MASK_DIR = "masks"  
# os.makedirs(MASK_DIR, exist_ok=True)

# def main():
#     # Load YOLOv8 model
#     model = YOLO('yolov8n.pt')  
#     model.to('cpu')  

#     # Counter for successful masks
#     mask_count = 0  

#     with open(IMAGE_LIST, 'r') as f:
#         image_paths = [line.strip() for line in f]

#     for img_path in image_paths:
#         results = model(img_path, verbose=False) 
#         boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []

#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Failed to read image: {img_path}")
#             continue  

#         h, w = img.shape[:2]

#         mask = np.zeros((h, w), dtype=np.uint8)
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=-1)

#         base_name = os.path.basename(img_path)
#         mask_name = os.path.splitext(base_name)[0] + "_mask.png"
#         mask_path = os.path.join(MASK_DIR, mask_name)
        
#         # Save mask only if it contains any bounding boxes
#         if np.any(mask):
#             cv2.imwrite(mask_path, mask)
#             mask_count += 1
#         else:
#             print(f"No objects detected for {img_path}")

#     print(f"Masks generated: {mask_count} out of {len(image_paths)} images.")

# if __name__ == "__main__":
#     main()
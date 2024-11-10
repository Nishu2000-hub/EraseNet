# EraseNet

This project focuses on developing a system for **object removal and scene reconstruction** using computer vision techniques, with an emphasis on image inpainting. The goal is to use pre-trained models to detect objects (e.g., cars, people) in images and then focus on creating a custom **inpainting solution** to reconstruct and fill the background where the objects once were. . The emphasis is on training an **auto-encoder from scratch** and experimenting with architectural innovations like **skip connections**, **dilated convolutions**, and varying network depths as the research component. This project is particularly useful for image editing and post-processing tasks where a clean background is required.

## Project Goals

- **Object Detection Using Pre-trained Models**: Utilize existing pre-trained object detection models (e.g., YOLO, Faster R-CNN) to identify objects in images.
- **Develop a Custom Inpainting Solution**: Focus on creating an inpainting model by training an auto-encoder from scratch using images.
- **Architectural Experimentation**: Investigate various auto-encoder architectures, including skip connections, dilated convolutions, and different network depths.
- **Scene Reconstruction**: Apply the custom inpainting model to reconstruct the background seamlessly after object removal in images.
- **Produce High-Quality Output**: Generate images with objects removed and backgrounds seamlessly reconstructed, suitable for applications like photo editing.

## Features

### 1. Object Detection (Using Pre-trained Models)

- **Purpose**: Accurately identify objects in images that need to be removed.
- **Implementation**:
  - Utilize pre-trained models like **YOLO** or **Faster R-CNN** for object detection.
  - This component provides input for the inpainting task but is not the primary focus.

### 2. Object Removal

- **Purpose**: Remove detected objects from images to create regions requiring inpainting.
- **Implementation**:
  - Apply masks over the detected objects to eliminate them from the images.
  - Prepare the masked images for the inpainting process.

### 3. Custom Inpainting Solution

- **Purpose**: Reconstruct and fill the background in the regions where objects have been removed.
- **Implementation**:
  - **Auto-Encoder Architecture**: Train an auto-encoder from scratch tailored for inpainting tasks.
  - **Architectural Innovations**:
    - **Skip Connections**: To preserve high-frequency details by allowing direct information flow between encoder and decoder layers.
    - **Dilated Convolutions**: To capture larger contextual information without increasing the model's size significantly.

### 4. Final Output

- **Purpose**: Produce seamless images with the objects removed and the background reconstructed convincingly.
- **Implementation**:
  - Combine the object detection, removal, and inpainting processes.
  - Generate high-quality images suitable for applications like photo editing and graphic design.

## Key Considerations

- **Focus on Inpainting**: the project emphasizes developing a robust inpainting solution as the core research component using images.
- **Use of Pre-trained Models for Object Detection**: Object detection is handled using existing models to allow more focus on the inpainting task.
- **Architectural Exploration in Inpainting**: Training the auto-encoder from scratch provides an opportunity to contribute original work.
- **Structural Features over Color**: The inpainting model will focus on structural features like edges and textures, making the solution robust to various lighting conditions and color variations.
- **No Facial Recognition**: The project does not involve recognizing or focusing on facial features.
- **Use of Static Images**: The shift from video frames to images simplifies the data and allows for more controlled experimentation.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Part 2: Data Acquisition and Preparation

### 1. Introduction

To train and evaluate the custom inpainting models, a comprehensive and diverse dataset is essential. The **Places2** dataset has been selected for its vast variety of scenes and suitability for image inpainting tasks.

### 2. Dataset Description

#### a. Source

- **Dataset**: Places2 Dataset (Places365-Standard)
- **Download Link**: http://places2.csail.mit.edu/download-private.html
- **Associated Paper**: Zhou, B., et al. "Places: A 10 million Image Database for Scene Recognition." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 40, no. 6, pp. 1452-1464, 2018.

#### b. Data Splitting Strategy

- **Training Set (60%)**: Approximately 1.08 million images.
- **Validation Set (20%)**: Approximately 360,000 images.
- **Test Set (20%)**: Approximately 360,000 images.

#### c. Differences Between Training and Validation Subsets

- **Scene Diversity**: Both subsets cover the same 365 scene categories but contain different images.
- **Complexity and Conditions**: Variations in lighting, weather conditions, and seasons are present in both sets but with different distributions.
- 
#### d. Number of Distinct Scenes and Samples

- **Scene Categories**: 365 distinct categories (e.g., beaches, forests, city streets, indoor rooms).
- **Samples per Category**: Thousands of images per category, ensuring ample data for learning diverse features.

#### e. Characterization of Samples

- **Resolution**: Images are standardized to **256×256 pixels** for consistency and computational efficiency.
- **Illumination and Ambient Conditions**: Includes images taken at different times of day, under various weather conditions, and across seasons.

## Methodology

### 1. Data Preparation

- **Image Organization**: Organize images into training, validation, and test sets.
- **Mask Generation**: Create masks to simulate object removal in images.
- **Normalization**: Scale pixel values to a standard range suitable for neural network processing.

### 2. Object Detection and Removal

- **Utilize Pre-trained Models**: Implement object detection using pre-trained models to identify objects in images.
- **Object Masking**: Create masks based on detected bounding boxes. Remove objects by applying these masks to the original images.

### 3. Inpainting and Scene Reconstruction

- **Developing the Auto-Encoder**:
  - **Baseline Model**: Start with a basic auto-encoder architecture.
  - **Incorporate Skip Connections**: Enhance the model's ability to reconstruct fine details.
  - **Implement Dilated Convolutions**: Allow the model to understand larger context without increasing parameters.
- **Training the Model**: Use the training set from the Places2 dataset. Train the auto-encoder to reconstruct the missing regions in images.


## Expected Outcomes

- **Effective Inpainting Solution**: A custom-trained auto-encoder capable of high-quality scene reconstruction after object removal in images.
- **Insights from Architectural Variations**: Understanding how skip connections, dilated convolutions, and network depths affect inpainting performance.
- **High-Quality Image Output**: Images where objects have been removed, and backgrounds are seamlessly reconstructed.

## Project Workflow

1. **Data Preparation**: Organize the Places2 dataset into training, validation, and test sets. Generate masks and feature extraction to simulate object removal in images.
2. **Object Detection**: Apply pre-trained models to detect objects in images.
3. **Object Removal**: Use detection results to mask and remove objects from images.
4. **Inpainting Model Development**: Build and train the auto-encoder with architectural innovations. Experiment with different configurations.
6. **Image Reconstruction**: Apply the trained model to reconstruct scenes in images. Generate final images with objects removed and backgrounds restored.


## Part 3: Data Preprocessing, Segmentation, and Feature Extraction

This section provides an overview of the data preprocessing and feature extraction steps implemented to prepare the images for the inpainting model. Each step aims to enhance the quality and usability of the dataset for the inpainting task.

### 1. Methods Applied for Data Preprocessing and Feature Extraction

The following steps were performed on each image in the dataset to ensure consistency and realism in the inpainting task.

#### Data Preprocessing
- **Resizing and Normalization**: Genrally the dataset Place365 given imaged as 256x256, but to be sure all images are resized to 256x256 pixels and normalized to a [0, 1] range to standardize inputs for the model and improve training stability.
- **YOLO-Based Object Detection and Masking**: YOLOv5, a pre-trained model, is used to detect objects within each image. Detected objects are masked (set to black) in each image, simulating object removal.

#### Feature Extraction
- **Canny Edge Detection**: This algorithm detects edges in the YOLO-masked images, providing the inpainting model with structural cues by highlighting edges and boundaries.
- **Contour Detection**: Contours are detected around masked areas, creating clear structural outlines. These outlines provide further spatial context for the inpainting model to interpret shapes and boundaries.

### 2. Justification of Methods

Each method was chosen for its specific contribution to the inpainting task, providing relevant structural and contextual information to guide the model as the other main thing will be traning on the auto encoder which has to be make from scratch , thats why YOLO has been used now.

#### Resizing and Normalization
- **Purpose**: Standardizing image dimensions (256x256) ensures consistency across inputs, while normalization scales pixel values to a [0, 1] range, stabilizing model training.
- **Justification**: Consistent dimensions and normalized values prevent variability that can disrupt training, improving the model’s learning efficiency.

#### YOLO-Based Object Detection and Masking
- **Purpose**: YOLO detects objects, allowing the model to focus on real-world scenarios by removing realistic, detected objects instead of applying random masks. whiuch give more accuracy on auto encoder part and connections.
- **Justification**: Using YOLO to detect and mask specific objects adds realism, presenting a meaningful challenge for the inpainting model to reconstruct real object regions.

#### Canny Edge Detection
- **Purpose**: Canny edge detection highlights structural details, giving the model cues about boundaries and object shapes around the masked areas.
- **Justification**: By focusing on edges, the inpainting model gains clues on where lines and shapes should be reconstructed, which is critical for realistic inpainting.

#### Contour Detection
- **Purpose**: Contour detection outlines object shapes and boundaries around the masked areas, enhancing spatial understanding for the inpainting model.
- **Justification**: Contours help the model recognize spatial relationships and structural integrity in the image, which is essential for accurate and realistic reconstruction.

### 3. Illustrations of Processing Steps

Below are sample illustrations of each step applied to the training data, demonstrating how each method prepares and processes images for inpainting.

#### Original Image
![Original Image]("C:\Users\khand\Places365_Dataset\train_100\00000002.jpg")

#### Resized and Normalized Image
![Resized and Normalized]("C:\Users\khand\Places365_Dataset\train_processed\resized_normalized\00000002.jpg")

#### YOLO Masked Image (Simulating Object Removal)
![YOLO Masked]("C:\Users\khand\Places365_Dataset\train_processed\yolo_masked\00000002.jpg")

#### Edge Detection
![Edge Detection]("C:\Users\khand\Places365_Dataset\train_processed\edge_detected\00000002.jpg")

#### Contour Detection
![Contour Detection]("C:\Users\khand\Places365_Dataset\train_processed\contour_detected\00000002.jpg")

These illustrations provide a visual overview of the transformations applied to each image in the dataset, from initial preprocessing to feature extraction.


### 4. Code and Instructions

The complete code for data preprocessing, segmentation, and feature extraction has been pushed to the repository. Below is a breakdown of each script and instructions to run them.

#### Code Overview
- `data_preprocessing_and_feature_extraction.py`: This script combines all data preprocessing and feature extraction steps, including resizing, normalization, YOLO-based masking, edge detection, and contour detection.

#### Instructions to Run Code

1. **Data Preprocessing and Feature Extraction**: Run `data_preprocessing_and_feature_extraction.py` to preprocess and extract features from images in each dataset split.
2. git clone https://github.com/ultralytics/yolov5 and install requirements -> pip install -r requirements.txt


 Places365_Dataset/
├── train_100/
│   ├── [category_1]/
│   │   ├── resized_normalized/
│   │   ├── yolo_masked/
│   │   ├── edge_detected/
│   │   └── contour_detected/
│   ├── [category_2]/
│   └── ...
├── val_100/
└── test_100/



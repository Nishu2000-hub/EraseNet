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

1. **Data Preparation**: Organize the Places2 dataset into training, validation, and test sets. Generate masks to simulate object removal in images.
2. **Object Detection**: Apply pre-trained models to detect objects in images.
3. **Object Removal**: Use detection results to mask and remove objects from images.
4. **Inpainting Model Development**: Build and train the auto-encoder with architectural innovations. Experiment with different configurations.
6. **Image Reconstruction**: Apply the trained model to reconstruct scenes in images. Generate final images with objects removed and backgrounds restored.

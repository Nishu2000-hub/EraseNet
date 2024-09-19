# EraseNet
This project focuses on **dynamic object removal and scene reconstruction** using computer vision techniques. The goal is to develop a system that processes input video footage, identifies moving objects (e.g., cars, people), and removes them from the scene. Once objects are removed, the system applies **inpainting methods** to reconstruct and fill the background where the objects once were. This project is particularly useful for video composition and post-editing tasks, where a clean background is required.

## Features
- **Object Detection:** Identify moving objects in video frames (e.g., cars, people).
- **Object Removal:** Remove the detected objects from each frame.
- **Inpainting:** Rebuild the removed portions of the scene using background estimation and inpainting techniques.
- **Final Output:** Generate video with and without objects, providing a seamless result for video composition

  ### Features to Focus On
The system will focus on calculating specific features to identify and remove objects, including:
1. **Bounding Boxes**: To mark the objects' location for removal.
2. **Contours, Edges, and Motion Vectors**: To help distinguish dynamic objects from the background based on their structural characteristics.

  ### Datasets
- **Object Detection Datasets**: To train the model for accurate identification and localization of moving objects.
- **Inpainting Datasets**: To ensure the model can accurately fill in the missing portions of the background, even in complex scenes.


## Project Goals
- Develop a system that processes video input and accurately **detects and removes moving objects**.
- Implement **background inpainting methods** to reconstruct the scene after object removal.
- Produce high-quality video results suitable for **post-production editing** and composition.


### Object Detection and Removal
The system will detect dynamic objects such as vehicles or pedestrians in the video frames using **bounding boxes** that localize the objects based on features such as shape and contours. Once an object is detected and localized, it will be "removed" by masking it out, which creates a gap in the scene that needs to be filled.



### Inpainting and Scene Reconstruction
After object removal, the system will apply **inpainting techniques** to fill the gaps left behind. The inpainting model will predict the missing areas by learning the surrounding textures, lines, and structures from the available visual context. 


The model will not rely on color or lighting conditions to detect objects,. The emphasis will be on detecting objects based on structural features like shape, movement, and contours, making it robust to various conditions.


### Key Considerations
- **No Facial Recognition**: Facial features will not be a focus of this project, as the solution will not include face recognition tasks.

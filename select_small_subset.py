
import os
import glob
import random

TRAIN_ROOT = r"C:\Users\khand\Places365_Dataset\train"
OUTPUT_TRAIN = "train_image_paths.txt"
OUTPUT_TEST = "test_image_paths.txt"
SUBSET_TRAIN_SIZE = 3000
SUBSET_TEST_SIZE = 600

def main():
    all_images = []
    for letter in os.listdir(TRAIN_ROOT):
        letter_path = os.path.join(TRAIN_ROOT, letter)
        if os.path.isdir(letter_path):
            for category in os.listdir(letter_path):
                category_path = os.path.join(letter_path, category)
                if os.path.isdir(category_path):
                    images = glob.glob(os.path.join(category_path, "*.jpg"))
                    all_images.extend(images)
    
    random.shuffle(all_images)
    train_subset = all_images[:SUBSET_TRAIN_SIZE]
    test_subset = all_images[SUBSET_TRAIN_SIZE:SUBSET_TRAIN_SIZE + SUBSET_TEST_SIZE]

    with open(OUTPUT_TRAIN, 'w') as f:
        for img in train_subset:
            f.write(img + "\n")

    with open(OUTPUT_TEST, 'w') as f:
        for img in test_subset:
            f.write(img + "\n")

    print(f"Selected {len(train_subset)} images for training.")
    print(f"Selected {len(test_subset)} images for testing.")

if __name__ == "__main__":
    main()





# import os
# import glob
# import random

# TRAIN_ROOT = r"C:\Users\khand\Places365_Dataset\train"
# OUTPUT_LIST = "train_image_paths.txt"
# SUBSET_SIZE = 3000  # Increased subset size for better training

# def main():
#     all_images = []
#     # Collect images from multiple categories
#     for letter in os.listdir(TRAIN_ROOT):  # Dynamically detect letters in dataset
#         letter_path = os.path.join(TRAIN_ROOT, letter)
#         if os.path.isdir(letter_path):
#             for category in os.listdir(letter_path):
#                 category_path = os.path.join(letter_path, category)
#                 if os.path.isdir(category_path):
#                     images = glob.glob(os.path.join(category_path, "*.jpg"))
#                     all_images.extend(images)
    
#     random.shuffle(all_images)
#     subset = all_images[:SUBSET_SIZE]

#     with open(OUTPUT_LIST, 'w') as f:
#         for img in subset:
#             f.write(img + "\n")
#     print(f"Selected {len(subset)} images for training.")

# if __name__ == "__main__":
#     main()

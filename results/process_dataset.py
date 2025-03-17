# -*- coding: utf-8 -*-
"""process_dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kJAD4croBZeX0EIgFa-YRuLBnfwfAABk
"""

import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# path
images_dir = "dataset/images"
xml_dir = "dataset/annotations"

current_dir = Path.cwd()

parent_dir = os.path.dirname(current_dir)

IMAGES_FOLDER = os.path.join(parent_dir, images_dir)
XML_FOLDER = os.path.join(parent_dir, xml_dir)

print("current_dir:", current_dir)
print("parent_dir:", parent_dir)
print("IMAGES_FOLDER:", IMAGES_FOLDER)
print("XML_FOLDER:", XML_FOLDER)

# read XML files
xml_files = [f for f in os.listdir(XML_FOLDER) if f.endswith(".xml")]

# read image files
image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.endswith(".png")]

print(len(xml_files))
print(len(image_files))

# keep xml and image file have same order
xml_files.sort()
image_files.sort()

def show_image(xml_path, image_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Count the number of each class
    class_counts = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
    bounding_boxes = []

    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Parse XML to get all bounding boxes
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_counts[class_name] += 1

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        bounding_boxes.append((class_name, xmin, ymin, xmax, ymax))

        # Draw bounding boxes
        color = (0, 255, 0) if class_name == "with_mask" else (255, 0, 0) if class_name == "without_mask" else (255, 165, 0)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Count the total number of bounding boxes
    total_boxes = sum(class_counts.values())

    # Display the image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Total Boxes: {total_boxes}\nWith Mask: {class_counts['with_mask']} | "
              f"Without Mask: {class_counts['without_mask']} | "
              f"Mask Weared Incorrect: {class_counts['mask_weared_incorrect']}")
    plt.show()

    # Print statistics
    print(total_boxes, class_counts)

xml_path = os.path.join(XML_FOLDER, xml_files[13])
image_path = os.path.join(IMAGES_FOLDER, image_files[13])

show_image(xml_path, image_path)

xml_path = os.path.join(XML_FOLDER, xml_files[0])
image_path = os.path.join(IMAGES_FOLDER, image_files[0])

show_image(xml_path, image_path)

# calculate number of each class and total number of boxes
class_counts = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
total_boxes = 0

for f in xml_files:
    xml_path = os.path.join(XML_FOLDER, f)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # all xml files
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_counts[class_name] += 1

print(class_counts)

total_boxes = sum(class_counts.values())

print(total_boxes)

"""
There are 853 images in our dataset, totally 4072 objects/faces.
"""


import os
import json
import xml.etree.ElementTree as ET
import random
import shutil
from sklearn.model_selection import train_test_split

# set the label map for the dataset
# 0: with_mask, 1: without_mask, 2: mask_weared_incorrect
label_map = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

# set the directories for images and annotations
current_path = os.getcwd()
DATASET_DIR = os.path.join(current_path, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
OUTPUT_JSON = os.path.join(DATASET_DIR,"dataset_coco.json")

# convert VOC XML annotations to COCO JSON format
def convert_voc_to_coco(images_dir, annotations_dir):
    images = []
    annotations = []
    categories = [{"id": i, "name": name, "supercategory": "object"} for name, i in label_map.items()]

    ann_id = 0
    img_id = 0

    xml_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith(".xml")])

    for xml_file in xml_files:
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in label_map:
                continue

            category_id = label_map[label]
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            w = xmax - xmin
            h = ymax - ymin

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    return coco_format


# dataset splits
# 60% train, 20% val, 20% test
SPLITS = ["train", "valid", "test"]
ANNOTATION_FILENAME = "_annotations.coco.json"
SPLIT_RATIOS = {"train": 0.6, "valid": 0.2, "test": 0.2}

def ensure_dirs():
    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

def split_dataset(coco):
    images = coco["images"]
    random.seed(42)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(SPLIT_RATIOS["train"] * n_total)
    n_valid = int(SPLIT_RATIOS["valid"] * n_total)

    return {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_valid],
        "test": images[n_train + n_valid:]
    }

def filter_annotations(annotations, image_ids):
    return [ann for ann in annotations if ann["image_id"] in image_ids]

def save_split(split_name, images, annotations, categories):
    split_dir = os.path.join(DATASET_DIR, split_name)

    # save _annotations.coco.json
    out_json_path = os.path.join(split_dir, ANNOTATION_FILENAME)
    with open(out_json_path, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, indent=2)

    # copy images
    for img in images:
        src = os.path.join(IMAGES_DIR, img["file_name"])
        dst = os.path.join(split_dir, img["file_name"])
        if not os.path.exists(dst):
            shutil.copy(src, dst)

def run():
    with open(OUTPUT_JSON, "r") as f:
        coco = json.load(f)

    ensure_dirs()
    split_images = split_dataset(coco)

    for split in SPLITS:
        images = split_images[split]
        image_ids = {img["id"] for img in images}
        annotations = filter_annotations(coco["annotations"], image_ids)
        save_split(split, images, annotations, coco["categories"])
        print(f"{split.title()} set: {len(images)} images")

if __name__ == "__main__":
    # convert the dataset to COCO format
    coco = convert_voc_to_coco(IMAGES_DIR, ANNOTATIONS_DIR)

    # save the dataset in COCO format
    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco, f, indent=2)

    # split the dataset into train, valid and test sets
    run()
    
# Train set: 511 images
# Valid set: 170 images
# Test set: 172 images
import json
import cv2
import os
import time
import argparse

def create_image_annotation(file_name, width, height, image_id):
    file_name = file_name.split('/')[-1]
    images = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images
    
def create_annotation_yolo_format(min_x, min_y, width, height, image_id, category_id, annotation_id):
    bbox = (min_x, min_y, width, height)
    area = round(width * height,2)

    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'category_id': category_id,
        'segmentation': []
    }

    return annotation

coco_format = {
    "images": [
        {
        }
    ],
    "categories": [

    ],
    "annotations": [
        {
        }
    ]
}

def images_annotations_info(yolo_path):

    path = []

    for subdir, dirs, files in os.walk(yolo_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                filepath = subdir+"/"+file
                path.append(filepath)

    annotations = []
    images = []

    image_id = 0
    annotation_id = 1   # In COCO dataset format, you must start annotation id with '1'

    for line in path:

        line = line.replace('\n', '')
        img_file = cv2.imread(line)

        # read a label file
        label_path = line[:-3]+"txt"
        label_file = open(label_path,"r")
        label_read_line = label_file.readlines()
        label_file.close()

        h, w, _ = img_file.shape

        # Create image annotation
        image = create_image_annotation(line, w, h, image_id)
        images.append(image)

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        for line1 in label_read_line:
            label_line = line1
            category_id = int(label_line.split()[0]) + 1    # you start with annotation id with '1'
            x_center = float(img_file.shape[1])*float(label_line.split()[1])
            y_center = float(img_file.shape[0])*float(label_line.split()[2])
            width = float(img_file.shape[1])*float(label_line.split()[3])
            height = float(img_file.shape[0])*float(label_line.split()[4])

            min_x = round(x_center-width/2,2)
            min_y = round(y_center-height/2,2)
            width = round(width,2)
            height = round(height,2)

            annotation = create_annotation_yolo_format(min_x, min_y, width, height, image_id, category_id, annotation_id)
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  # if you finished annotation work, updates the image id.

    return images, annotations

def create_cocofromyolo(yolo_path, labels_path, output_name):
    output_path = output_name + '.json'
    coco_format['images'], coco_format['annotations'] = images_annotations_info(yolo_path)
    classes = open(label_path).read().strip().split("\n")
    for index, label in enumerate(classes):
        ann = {
            "supercategory": "detection_objects",
            "id": index + 1,  # Index starts with '1' .
            "name": label
        }
        coco_format['categories'].append(ann)

    with open(output_path, 'w') as outfile:
        json.dump(coco_format, outfile)

def get_args():
    parser = argparse.ArgumentParser('Yolo annotations to COCO annotation converter helper')
    parser.add_argument('-p', '--path', type=str, required=True, help='(Absolute) path for folder containing image files and yolo annotated txt files')
    parser.add_argument('-l', '--labels', type=str, required=True ,help='(Absolute) path to file containing objection detection category names')
    parser.add_argument('-o', '--output', default="output", type=str, help='Name of the output json file')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start = time.time()
    opt = get_args()
    yolo_path = opt.path
    label_path = opt.labels
    output_name = opt.output
    create_cocofromyolo(yolo_path, label_path, output_name)
    print("YOLO annotation converted to COCO annotations in: " + output_name+".json")
    print("Conversion processed in " + str(float(time.time()-start)) + " seconds")
import cv2
import sys
import os
import json
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import argparse
import time

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.size = None
        self.filename = ""
        self.filepath = filepath
        self.verified = False
        self.parseXML()

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox, difficult):
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult))

    def addSize(self, width, height, depth):
        self.size = ((width,height,depth))

    def getSize(self):
        return self.size

    def getFilename(self):
        return self.filename

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser()
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        self.filename = xmltree.find('filename').text

        for object_iter in xmltree.findall('size'):
            width = int(object_iter.find("width").text)
            height = int(object_iter.find("height").text)
            depth = int(object_iter.find("depth").text)
            self.addSize(width,height,depth)
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text

            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.addShape(label, bndbox, difficult)
        return True

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

def images_annotations_info(vocfolder_path, classes):
    annotations = []
    images = []

    image_id = 0
    annotation_id = 1 
    for file in os.listdir(vocfolder_path):
        if file.endswith(".xml"):
            
            filePath = vocfolder_path + "/" + file
            VocParseReader = PascalVocReader(filePath)
            size = VocParseReader.getSize()
            filename = VocParseReader.getFilename()
            image = create_image_annotation(filename, size[0], size[1], image_id)
            images.append(image)

            shapes = VocParseReader.getShapes()
            num_of_box = len(shapes)

            for i in range(num_of_box):
                category_id = classes.index(shapes[i][0]) + 1
                min_x = shapes[i][1][0][0]
                min_y = shapes[i][1][0][1]
                max_x = shapes[i][1][2][0]
                max_y = shapes[i][1][2][1]

                width = round(max_x - min_x,2)
                height = round(max_y - min_y,2)

                annotation = create_annotation_yolo_format(min_x, min_y, width, height, image_id, category_id, annotation_id)
                annotations.append(annotation)
                annotation_id += 1

            image_id += 1

    return images,annotations

def create_cocofrompascal(vocfolder_path, label_path, output_name):
    output_path = output_name + '.json'
    classes = open(label_path).read().strip().split("\n")
    coco_format['images'], coco_format['annotations'] = images_annotations_info(vocfolder_path, classes)
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
    parser = argparse.ArgumentParser('PascalVOC annotations to COCO annotation converter helper (Assumes that image directory is already present)')
    parser.add_argument('-v', '--voc', type=str, required=True, help='(Absolute) path for folder containing the PascalVOC VML files')
    parser.add_argument('-l', '--labels', type=str, required=True ,help='(Absolute) path to file containing objection detection category names')
    parser.add_argument('-o', '--output', default="output", type=str, help='Name of the output json file')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start = time.time()
    opt = get_args()
    vocfolder_path = opt.voc
    label_path = opt.labels
    output_name = opt.output
    create_cocofrompascal(vocfolder_path, label_path, output_name)
    print("PascalVOC annotation converted to COCO annotations in: " + output_name+".json")
    print("Conversion processed in " + str(float(time.time()-start)) + " seconds")
import cv2
import sys
import os
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import argparse
import time

XML_EXT = '.xml'
TXT_EXT = '.txt'
ENCODE_METHOD = 'utf-8'

class YOLOWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def BndBox2YoloLine(self, box, classList=[]):
        xmin = box['xmin']
        xmax = box['xmax']
        ymin = box['ymin']
        ymax = box['ymax']

        xcen = float((xmin + xmax)) / 2 / self.imgSize[1]
        ycen = float((ymin + ymax)) / 2 / self.imgSize[0]

        w = float((xmax - xmin)) / self.imgSize[1]
        h = float((ymax - ymin)) / self.imgSize[0]

        # PR387
        classIndex = box['name']

        return classIndex, xcen, ycen, w, h

    def save(self, classList=[], targetFile=None):

        out_file = None 

        if targetFile is None:
            out_file = open(
            self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)

        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)


        for box in self.boxlist:
            classIndex, xcen, ycen, w, h = self.BndBox2YoloLine(box, classList)
            out_file.write("%d %.6f %.6f %.6f %.6f\n" % (classIndex, xcen, ycen, w, h))

        out_file.close()

class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
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

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser()
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
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

def create_yolofrompascal(imagefolder_path, vocfolder_path, label_path):

    for file in os.listdir(vocfolder_path):
        if file.endswith(".xml"):
            
            annotation_no_xml = os.path.splitext(file)[0]

            imagePath = os.path.join(imagefolder_path, annotation_no_xml + ".jpg")

            image = cv2.imread(imagePath)
            imgFolderName = os.path.basename(imagefolder_path)
            imgFileName = os.path.basename(imagePath)

            writer = YOLOWriter(imgFolderName, imgFileName, image.shape, localImgPath=imagePath)

            # Read classes.txt
            classListPath = label_path
            classesFile = open(classListPath, 'r')
            classes = classesFile.read().strip('\n').split('\n')
            classesFile.close()

            # Read VOC file
            filePath = vocfolder_path + "/" + file
            VocParseReader = PascalVocReader(filePath)
            shapes = VocParseReader.getShapes()
            num_of_box = len(shapes)

            for i in range(num_of_box):
                label = classes.index(shapes[i][0])
                xmin = shapes[i][1][0][0]
                ymin = shapes[i][1][0][1]
                x_max = shapes[i][1][2][0]
                y_max = shapes[i][1][2][1]

                writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

            writer.save(targetFile= imagefolder_path + "/" + annotation_no_xml + ".txt")

def get_args():
    parser = argparse.ArgumentParser('PascalVOC annotations to YOLO annotation converter helper')
    parser.add_argument('-p', '--path', type=str, required=True, help='(Absolute) path for folder containing image files')
    parser.add_argument('-v', '--voc', type=str, required=True, help='(Absolute) path for folder containing the PascalVOC VML files')
    parser.add_argument('-l', '--labels', type=str, required=True ,help='(Absolute) path to file containing objection detection category names')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start = time.time()
    opt = get_args()
    imagefolder_path = opt.path
    vocfolder_path = opt.voc
    label_path = opt.labels
    create_yolofrompascal(imagefolder_path, vocfolder_path, label_path)
    print("PascalVOC annotation converted to YOLO annotations in: " + imagefolder_path+" folder")
    print("Conversion processed in " + str(float(time.time()-start)) + " seconds")
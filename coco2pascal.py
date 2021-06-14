from pycocotools.coco import COCO
import requests
import os
import argparse
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import cv2
import time

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='detection_objects', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(each_object['ymax']) == int(self.imgSize[0]) or (int(each_object['ymin'])== 1):
                truncated.text = "1" # max == height or min
            elif (int(each_object['xmax'])==int(self.imgSize[1])) or (int(each_object['xmin'])== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

def create_labelsfile(coco):
    cats = coco.loadCats(coco.getCatIds())
    classes = [cat['name'] for cat in cats]
    filename = "obj.names"
    with open(filename, 'w') as f:
        for cat in classes:
            f.write("%s\n" % cat)
    f.close()
    return classes

def create_pascalfromcoco(imagefolder_path, cocojson_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    coco = COCO(cocojson_path)
    create_labelsfile(coco)
    imgIds = coco.getImgIds()
    for i in imgIds:
        img = coco.loadImgs(i)[0]
        imgFileName = img['file_name']
        imagePath = os.path.join(imagefolder_path, imgFileName)
        annotation_no_txt = os.path.splitext(imgFileName)[0]
        imgFolderName = os.path.basename(imagefolder_path)

        image = cv2.imread(imagePath)
        writer = PascalVocWriter(imgFolderName, imgFileName, image.shape, localImgPath=imagePath)

        annIds = coco.getAnnIds(imgIds=i)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            label = coco.loadCats(ann["category_id"])[0]['name']
            bbox = ann['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            x_max = bbox[2] + bbox[0]
            y_max = bbox[3] + bbox[1]

            writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

        writer.save(targetFile= output_directory+ "/" + annotation_no_txt + ".xml")

def get_args():
    parser = argparse.ArgumentParser('COCO annotations to ParscalVOC annotation converter helper')
    parser.add_argument('-p', '--path', type=str, required=True, help='(Absolute) path for folder containing image files')
    parser.add_argument('-j', '--json', type=str, required=True ,help='(Absolute) path to COCO annotated json file')
    parser.add_argument('-o', '--output', default="pascal", type=str, help='Name of the directory to store PascalVOC VML files')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start = time.time()
    opt = get_args()
    imagefolder_path = opt.path
    cocojson_path = opt.json
    output_directory = opt.output
    create_pascalfromcoco(imagefolder_path, cocojson_path, output_directory)
    print("COCO annotation converted to PascalVOC annotations in: " + output_directory+" folder")
    print("Conversion processed in " + str(float(time.time()-start)) + " seconds")
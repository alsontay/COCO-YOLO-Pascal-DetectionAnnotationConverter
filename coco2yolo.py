from pycocotools.coco import COCO
import requests
import os
import argparse
import time

def create_labelsfile(coco):
    cats = coco.loadCats(coco.getCatIds())
    classes = [cat['name'] for cat in cats]
    filename = "obj.names"
    with open(filename, 'w') as f:
        for cat in classes:
            f.write("%s\n" % cat)
    f.close()
    return classes

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def create_yolofromcoco(cocojson_path, imagefolder_path):
    coco = COCO(cocojson_path)
    classes = create_labelsfile(coco)
    for cat in classes:
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)
        images = coco.loadImgs(imgIds)
        for im in images:
            dw = 1. / im['width']
            dh = 1. / im['height']

            annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            
            filename = im['file_name'].replace(".jpg", ".txt")
            with open(imagefolder_path + "/" + filename, "a") as f:
                for i in range(len(anns)):
                    xmin = anns[i]["bbox"][0]
                    ymin = anns[i]["bbox"][1]
                    xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
                    ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]
                    
                    x = (xmin + xmax)/2
                    y = (ymin + ymax)/2
                    
                    w = xmax - xmin
                    h = ymax-ymin
                    
                    x = x * dw
                    w = w * dw
                    y = y * dh
                    h = h * dh
                    catno = catIds[0]-1
                    mystring = str(str(catno)+" " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
                    f.write(mystring)
                    f.write("\n")

            f.close()

def get_args():
    parser = argparse.ArgumentParser('COCO annotations to YOLO annotation converter helper')
    parser.add_argument('-p', '--path', type=str, required=True, help='(Absolute) path for folder containing image files')
    parser.add_argument('-j', '--json', type=str, required=True ,help='(Absolute) path to COCO annotated json file')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start = time.time()
    opt = get_args()
    imagefolder_path = opt.path
    cocojson_path = opt.json
    create_yolofromcoco(cocojson_path, imagefolder_path)
    print("COCO annotation converted to YOLO annotations in: " + imagefolder_path+" folder")
    print("Conversion processed in " + str(float(time.time()-start)) + " seconds")
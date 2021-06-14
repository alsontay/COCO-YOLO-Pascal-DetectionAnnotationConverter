### Image Detection Annotation Converter

Performs conversion between YOLO, PASCAL VOC and COCO annotation format.

### Format and Usage:
- Images must be in .jpg/.png format and reside within an independent folder.
- COCO annotations exists as a single `.json` file within the directory.
- YOLO annotations exists as `.txt` files within the same folder as images, having the same names as each the image.
- PASCAL VOC annotations exists as `.xml` files in a separate independent folder, having the same names as each image.
- Conversions from PASCAL VOC and YOLO will require an `obj.names` text file in the directory that contains the names of all detection categories (one category per line).
- Conversions from COCO will create an `obj.names` text file in the directory that contains the names of all detection categories (one category per line).
- Each conversion script will require different arguments as inputs (refer to example) and will output the converted annotation to the correct format.
- Do note that the conversions are only accurate to .2dp, hence mulitple conversions may not result in same outputs.
### Requirements:
- python
- argparse
- xml-python
- opencv-python
- pycocotools

 These packages by using pip in command line, assuming python is already installed:
  - `pip install argparse`
  - `pip install xml-python`
  - `pip install opencv-python`
  - `pip install pycocotools`
### Example:
- Make sure your command line is in the correct directory containing the scripts, and your images and pre-converted annotations are in the correct format.
- Scripts are named according to which format you want to convert from and to.
  - YOLO to COCO (yolo2coco.py)
    - ```yolo2coco.py -p images -l obj.names -o output```
    - Running the above line in cmd should generate an `output.json` file in COCO format.
  - Pascal to COCO (pascal2coco.py)
    - ```pascal2coco.py -v pascal -l obj.names -o output```
    - Running the above line in cmd should generate an `output.json` file in COCO format.
  - Pascal to YOLO (pascal2yolo.py)
    - ```pascal2yolo.py -p images -l obj.names -v pascal```
    - Running the above line in cmd should generate `.txt` files in the image directory in YOLO format.
  - COCO to YOLO (coco2yolo.py)
    - ```coco2yolo.py -p images -j output.json```
    - Running the above line in cmd should generate `.txt` files in the image directory in YOLO format.
  - COCO to Pascal (coco2pascal.py)
    - ```coco2pascal.py -p images -j output.json -o pascal```
    - Running the above line in cmd should generate a directory named `pascal` that contains `.xml` files in PASCAL VOC format.
  - YOLO to Pascal (yolo2pascal.py)
    - ```yolo2pascal.py -p images -l obj.names -o pascal```
    - Running the above line in cmd should generate a directory named `pascal` that contains `.xml` files in PASCAL VOC format.
- Use `"scriptname.py" -h` to get help on each of the required and optional arguments and inputs.
### Credits:
 - Scripts involving Pascal is adapted from [hai-h-nguyen/Yolo2Pascal-annotation-conversion](https://github.com/hai-h-nguyen/Yolo2Pascal-annotation-conversion)

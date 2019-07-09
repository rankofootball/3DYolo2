# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from base64 import b64encode, b64decode
from pascal_voc_io import PascalVocWriter
import os.path
import sys

class LabelFileError(Exception):
    pass

class LabelFile(object):
    # It might be changed as window creates
    suffix = '.lif'

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)

    def savePascalVocFormat(self, filename, shapes, imagePath, imageData,
            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # imageData contains x,y,z, depth extensions
        imageShape = [imageData[0], imageData[1], imageData[2], imageData[3] ]
        writer = PascalVocWriter(imgFolderName, imgFileNameWithoutExt+".tif",\
                                 imageShape, localImgPath=imagePath)
        bSave = False
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            bndbox = LabelFile.convertPoints2BndBox(points)
            writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], bndbox[4], bndbox[5], label)
            bSave = True

        if bSave:
            writer.save(targetFile = filename)
        return

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        zmin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        zmax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            z = p[2]
            xmin = min(x,xmin)
            ymin = min(y,ymin)
            zmin = min(z,zmin)
            xmax = max(x,xmax)
            ymax = max(y,ymax)
            zmax = max(z,zmax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if (xmin < 1):
            xmin = 1

        if (ymin < 1):
            ymin = 1

        return (int(xmin), int(ymin), int(zmin), int(xmax), int(ymax), int(zmax))

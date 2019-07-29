from skimage import io
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv3D, Input, MaxPooling3D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
#import imgaug as ia
#from tqdm import tqdm
#from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2
from preprocessing3D import parse_annotation, BatchGenerator
from utils3D import WeightReader, decode_netout, draw_boxes

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf

filename_annot = "/Users/sten/Documents/SpineImages/Dendrites_3D/tests/image.xml"
filename_image = "/Users/sten/Documents/SpineImages/Dendrites_3D/images_train/image75.tif"



LABELS = ['spine']

IMAGE_Z, IMAGE_H, IMAGE_W = 96, 416, 416    # when changing the 96 change as well in preproceessing3d and utils3d
GRID_Z,  GRID_H, GRID_W   = 3, 13, 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.2#0.5
NMS_THRESHOLD    = 0.2#0.45
ANCHORS          = [0.57273, 0.677385, 0.5, 1.87446, 2.06253, 2.0, 3.33843, 5.47434, 3.0, 7.88282, 3.52778, 1.0, 9.77052, 9.16828, 9.0]

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 1  # 16
WARM_UP_BATCHES  = 1
TRUE_BOX_BUFFER  = 50




#wt_path = '/users/tsd/sten/yolotest/keras-yolo2/Weights/yolov2.weights'                     
#train_image_folder = '/users/tsd/sten/yolotest/keras-yolo2/images3d/images_train/'
#train_annot_folder = '/users/tsd/sten/yolotest/keras-yolo2/images3d/annot_train/'
#valid_image_folder = '/users/tsd/sten/yolotest/keras-yolo2/images3d/images_val/'
#valid_annot_folder = '/users/tsd/sten/yolotest/keras-yolo2/images3d/annot_val/'
#wt_path = '/home/sten/yolo/yolov2.weights'                     
#train_image_folder = '/home/sten/yolo/images3d/images_train/'
#train_annot_folder = '/home/sten/yolo/images3d/annot_train/'
#valid_image_folder = '/home/sten/yolo/images3d/images_val/'
#valid_annot_folder = '/home/sten/yolo/images3d/annot_val/'


from pascal_voc_io import PascalVocReader
#from shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
#from PyQt4.QtGui import *
#from PyQt4.QtCore import *
from labelFile import *
#itemsToShapes = {}
#shapesToItems = {}
usingPascalVocFormat = True


def saveLabels(filename,boxes):
    lf = LabelFile()
    def format_shape(s):
        s.xmin *= IMAGE_W
        s.ymin *= IMAGE_H
        s.zmin *= IMAGE_Z
        s.xmax *= IMAGE_W
        s.ymax *= IMAGE_H
        s.zmax *= IMAGE_Z
        return dict(label='spine', points=[(s.xmin, s.ymin, s.zmin),(s.xmax, s.ymax, s.zmax) ], line_color=None, fill_color=None)

#    print ("objbox = ", objbox)
    shapes = [format_shape(shape) for shape in boxes]
    # Can add differrent annotation formats here
    try:
        if usingPascalVocFormat is True:
            print ('savePascalVocFormat save to:' + filename)
            lf.savePascalVocFormat(filename, shapes, filename, [IMAGE_W, IMAGE_H, IMAGE_Z, 1 ], None, None)
        else:
            lf.save(filename, shapes, unicode(filename), self.lineColor.getRgb(), self.fillColor.getRgb())
            self.labelFile = lf
            self.filename = filename
        return True
    except LabelFileError as e:
        self.errorMessage(u'Error saving label data',
                u'<b>%s</b>' % e)
        return False



# CONSTRUCT THE NETOWRK



# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(input):
    block_size = 2
    block_size_3 = block_size*block_size*block_size

    batch_size = tf.shape(input)[0]
    s_height = 2*GRID_H #tf.shape(input)[1]
    s_width = 2*GRID_W# tf.shape(input)[2] 
    s_z = 2*GRID_Z# tf.shape(input)[3]
    s_depth = tf.shape(input)[4]

 #   blocksize= tf.Variable(2.0, tf.float32)
    d_depth = tf.math.scalar_mul(block_size_3,s_depth)
    d_width = GRID_W #tf.math.scalar_mul(1/block_size,s_width)
    d_height = GRID_H #tf.math.scalar_mul(1/block_size,s_height)
    d_z = GRID_Z #tf.math.scalar_mul(1/block_size,s_z) 

#    y=tf.reshape(input,[batch_size, d_z, block_size,d_height, block_size, d_width, block_size,s_depth])
#    y=tf.transpose(y, [0,1,2,3,4,5,7,6])

    y=tf.reshape(input,[batch_size, d_z, block_size,d_height, block_size, d_width, block_size*s_depth])
    y=tf.transpose(y, [0,1,2,3,5,4,6])
    y=tf.reshape(input,[batch_size, d_z, block_size,d_height, d_width, block_size*block_size*s_depth])
    y=tf.transpose(y, [0,1,3,4,2,5])
    y=tf.reshape(y,[batch_size, d_z, d_height, d_width, 512])
    return y



#def space_to_depth_x2(x):
#    return tf.nn.space_to_depth(x, block_size=2)

input_image = Input(shape=(IMAGE_Z, IMAGE_H, IMAGE_W, 1))
true_boxes  = Input(shape=(1, 1, 1, 1, TRUE_BOX_BUFFER , 6))  # ?,?,?,  #, x,y,z,h,w,d 


# Layer 1
x = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)

# Layer 2
x = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(2, 2,2))(x)

# Layer 3
x = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv3D(64, (1,1,1), strides=(1,1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)

# Layer 6
x = Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv3D(128, (1,1,1), strides=(1,1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(2, 2,2))(x)

# Layer 9
x = Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv3D(256, (1,1,1), strides=(1,1,1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv3D(256, (1,1,1), strides=(1,1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv3D(512, (3,3,3), strides=(1,1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling3D(pool_size=(2, 2,2))(x)

# Layer 14
x = Conv3D(1024, (3,3,3), strides=(1,1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv3D(512, (1,1,1), strides=(1,1,1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv3D(1024, (3,3,3), strides=(1,1,1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv3D(512, (1,1,1), strides=(1,1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv3D(1024, (3,3,3), strides=(1,1,1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv3D(1024, (3,3,3), strides=(1,1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv3D(1024, (3,3,3), strides=(1,1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv3D(64, (1,1,1), strides=(1,1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv3D(1024, (3,3,3), strides=(1,1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
#x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
x = Conv3D(BOX * (6 + 1 + CLASS), (1,1,1), strides=(1,1,1), padding='same', name='conv_23')(x)
output = Reshape((GRID_Z, GRID_H, GRID_W, BOX, 6 + 1 + CLASS))(x)

# small hack to allow true_boxes to be registered when Keras build the model 
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

model.summary()



model.load_weights("/Users/sten/Documents/SpineImages/weights_spine_3D.h5")

image = io.imread(filename_image)[:,:,:,0]
image = image.reshape((96,416,416,1))

# image = cv2.imread('images3d/images_train/image8.tif')
dummy_array = np.zeros((1,1,1,1,1,TRUE_BOX_BUFFER,6))

# plt.figure(figsize=(10,10))

#input_image = cv2.resize(image, (96, 416, 416))
image = np.array(image)
input_image = np.divide(image,255.0)
#input_image = input_image[:,:,:,::-1]
input_image = np.expand_dims(input_image, 0)


print (input_image.shape)

netout = model.predict([input_image, dummy_array])

#print (netout[0])
boxes = decode_netout(netout[0], 
                      obj_threshold=OBJ_THRESHOLD,
                      nms_threshold=NMS_THRESHOLD,
                      anchors=ANCHORS, 
                      nb_class=CLASS)


            
for nb in range(len(boxes)):
    print (boxes[nb].xmin,boxes[nb].xmax,boxes[nb].ymin,boxes[nb].ymax,boxes[nb].zmin,boxes[nb].zmax,boxes[nb].c)
#image = draw_boxes(image, boxes, labels=LABELS)

saveLabels(filename_annot,boxes)



#plt.imshow(image[:,:,::-1]); plt.show()
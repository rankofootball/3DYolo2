from keras.models import Sequential, Model
from keras.layers import Dropout, Reshape, Activation, Conv3D, Input, MaxPooling3D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
#import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2
from preprocessing3D import parse_annotation, BatchGenerator
from utils3D import WeightReader, decode_netout, draw_boxes

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import tensorflow as tf


# include large model support
from tensorflow_large_model_support import LMSKerasCallback 
# LMSKerasCallback and LMS share a set of keyword arguments. Here we just
# use the default options.
lms_callback = LMSKerasCallback()




LABELS = ['spine']

IMAGE_Z, IMAGE_H, IMAGE_W = 96, 416, 416    # when changing the 96 change as well in preproceessing3d and utils3d
GRID_Z,  GRID_H, GRID_W   = 3, 13, 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3#0.5
NMS_THRESHOLD    = 0.3#0.45
ANCHORS          = [0.4, 0.4, 0.4,    0.6, 0.6, 0.6,    0.8, 0.4, 0.4,    0.4, 0.8, 0.4,    0.4, 0.4, 0.8]
#ANCHORS          = [0.4, 0.4, 0.4,    0.6, 0.6, 0.6]

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 1  # 16
WARM_UP_BATCHES  = 4
TRUE_BOX_BUFFER  = 50




#wt_path = '/users/tsd/sten/yolotest/keras-yolo2/Weights/yolov2.weights'                     
#train_image_folder = '/users/tsd/sten/yolotest/keras-yolo2/images3d/images_train/'
#train_annot_folder = '/users/tsd/sten/yolotest/keras-yolo2/images3d/annot_train/'
#valid_image_folder = '/users/tsd/sten/yolotest/keras-yolo2/images3d/images_val/'
#valid_annot_folder = '/users/tsd/sten/yolotest/keras-yolo2/images3d/annot_val/'
wt_path = '/u/sten/yolo/yolov2.weights'                     
train_image_folder = '/u/sten/3dyolo/images3d/images_train/'
train_annot_folder = '/u/sten/3dyolo/images3d/annot_train/'
valid_image_folder = '/u/sten/3dyolo/images3d/images_val/'
valid_annot_folder = '/u/sten/3dyolo/images3d/annot_val/'


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

    y=tf.reshape(input,[batch_size, d_z, block_size,d_height, block_size, d_width, block_size*s_depth])
    y=tf.transpose(y, [0,1,2,3,5,4,6])
    y=tf.reshape(y,[batch_size, d_z, block_size,d_height, d_width, block_size*block_size*s_depth])
    y=tf.transpose(y, [0,1,3,4,2,5])
    y=tf.reshape(y,[batch_size, d_z, d_height, d_width, 512])
    return y



#def space_to_depth_x2(x):
#    return tf.nn.space_to_depth(x, block_size=2)

input_image = Input(shape=(IMAGE_Z, IMAGE_H, IMAGE_W, 1))
true_boxes  = Input(shape=(1, 1, 1, 1, TRUE_BOX_BUFFER , 6))  # ?,?,?,  #, x,y,z,h,w,d 
dropout_rate = 0.2

# Layer 1
x = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)
Dropout(dropout_rate)(x)

# Layer 2
x = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(2, 2,2))(x)
Dropout(dropout_rate)(x)

# Layer 3
x = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)
Dropout(dropout_rate)(x)

# Layer 4
x = Conv3D(64, (1,1,1), strides=(1,1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)
Dropout(dropout_rate)(x)

# Layer 5
x = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)
Dropout(dropout_rate)(x)

# Layer 6
x = Conv3D(256, (3,3,3), strides=(1,1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)
Dropout(dropout_rate)(x)

# Layer 7
x = Conv3D(128, (1,1,1), strides=(1,1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)
Dropout(dropout_rate)(x)

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

model.summary(positions=[0.2,0.5,0.6,0.8,1.0])


#----------------------------------------------------------------------------
#Load pretrained weights
#Load the weights originally provided by YOLO
#weight_reader = WeightReader(wt_path)


#weight_reader.reset()
#nb_conv = 23

#for i in range(1, nb_conv+1):
#    conv_layer = model.get_layer('conv_' + str(i))
#    
#    if i < nb_conv:
#        norm_layer = model.get_layer('norm_' + str(i))
#        
#        size = np.prod(norm_layer.get_weights()[0].shape)
#
#        beta  = weight_reader.read_bytes(size)
#        gamma = weight_reader.read_bytes(size)
#        mean  = weight_reader.read_bytes(size)
#        var   = weight_reader.read_bytes(size)
#
#        weights = norm_layer.set_weights([gamma, beta, mean, var])       
#        
#    if len(conv_layer.get_weights()) > 1:
#        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
#        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
#        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
#        kernel = kernel.transpose([2,3,1,0])
#        conv_layer.set_weights([kernel, bias])
#    else:
#        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
#        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
#        kernel = kernel.transpose([2,3,1,0])
#        conv_layer.set_weights([kernel])


# Randomize weights of all conv layers

#nb_conv = 23
#for i in range(1, nb_conv+1):
#    conv_layer = model.get_layer('conv_' + str(i))
#    
#    weights = conv_layer.get_weights()
#    if len(conv_layer.get_weights()) > 1:
#        kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W*GRID_Z)
#        bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W*GRID_Z)
#        conv_layer.set_weights([kernel, bias])
#    else:
#        kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W*GRID_Z)
#        conv_layer.set_weights([kernel])        


# Randomize weights of the last layer

#layer   = model.layers[-4] # the last convolutional layer
#weights = layer.get_weights()

#new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W*GRID_Z)
#new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W*GRID_Z)

#layer.set_weights([new_kernel, new_bias])

#--------------------------------------------------------------------------------

#continue with saved weight matrix

model.load_weights("weights_spine_3D.h5") 

#Perform training


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:5]
#    mask_shape = tf.shape(y_true)[:4]

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H*GRID_Z]),(1, GRID_Z, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_H), [GRID_W*GRID_Z]),(1, GRID_Z, GRID_W, GRID_H, 1, 1)))
    cell_y = tf.transpose(cell_y,(0,1,3,2,4,5))
    cell_z = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_Z), [GRID_H*GRID_W]),(1, GRID_W, GRID_H, GRID_Z, 1, 1)))
    cell_z = tf.transpose(cell_z,(0,3,2,1,4,5))
    
#    cell_y = tf.transpose(cell_x, (0,2,1,3,4,5))
#    cell_z = tf.transpose(cell_x, (0,3,2,1,4,5))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y,cell_z], -1), [BATCH_SIZE, 1, 1, 1, BOX , 1])
 #   cell_grid = tf.tile(tf.concat([cell_x,cell_y,cell_z], -1), [BATCH_SIZE, 1, 1, 5, 1])
    
    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)
    
    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    
    """
    Adjust prediction
    """
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :3]) + cell_grid
    
    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 3:6]) * np.reshape(ANCHORS, [1,1,1,1,BOX,3])
    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 6])
    
    ### adjust class probabilities
    pred_box_class = y_pred[..., 7:]
    
    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:3] # relative position to the containing cell
   
    ### adjust w and h
    true_box_wh = y_true[..., 3:6] # number of cells accross, horizontally and vertically
    
    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half
    
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] * intersect_wh[..., 2]
    
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1] * true_box_wh[..., 2] 
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1] * pred_box_wh[..., 2] 
#    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1] 
#    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    user5 = iou_scores[0,0,6,4,...]
    user6 = y_pred[0,0,6,4,...,6]
    true_box_conf = iou_scores * y_true[..., 6]

    ### adjust class probabilities
    true_box_class = tf.to_int32(0 *  y_true[..., 6])
#    true_box_class = tf.argmax(y_true[..., 7:], -1)     # original: get index of maximal value over all classes
    
    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 6], axis=-1) * COORD_SCALE
    
    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:3]
    true_wh = true_boxes[..., 3:6]
    
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_xy = tf.expand_dims(pred_box_xy, 5)
    pred_wh = tf.expand_dims(pred_box_wh, 5)
    
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] * intersect_wh[..., 2] 
    
    true_areas = true_wh[..., 0] * true_wh[..., 1] * true_wh[..., 2]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1] * pred_wh[..., 2]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=5)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 6]) * NO_OBJECT_SCALE
    
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 6] * OBJECT_SCALE
    
    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 6] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
    
    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
    seen = tf.assign_add(seen, 1.)

    

    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,1,BOX,3]) * no_boxes_mask, 
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy, 
                                   true_box_wh,
                                   coord_mask])
   
    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    


    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh_pred    = tf.reduce_sum(tf.square(pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh_true    = tf.reduce_sum(tf.square(true_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
    loss = loss_xy + loss_wh + loss_conf + loss_class
    
    nb_true_box = tf.reduce_sum(y_true[..., 6])
    user1 = y_true[0,0,6,4,...,2]
    user2 = pred_box_xy[0,0,6,4,...,2]
    user3 = nb_coord_box
    user4 = nb_conf_box   

    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """   
    sess = K.get_session()
    sess.run(tf.contrib.memory_stats.BytesInUse())

    current_recall = nb_pred_box/(nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall) 

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh_pred], message='Loss WH pred\t', summarize=1000)
    loss = tf.Print(loss, [loss_wh_true], message='Loss WH true\t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    

    loss = tf.Print(loss, [user1], message='user1 \t', summarize=1000)
    loss = tf.Print(loss, [user2], message='user2 \t', summarize=1000)
    loss = tf.Print(loss, [user3], message='nb coord box \t', summarize=1000)
    loss = tf.Print(loss, [user4], message='nb conf box \t', summarize=1000)
    loss = tf.Print(loss, [user5], message='iou \t', summarize=1000)
    loss = tf.Print(loss, [user6], message='y_pred 6 \t', summarize=1000)
    return loss


#Parse the annotations to construct train generator and validation generator


generator_config = {
    'IMAGE_Z'         : IMAGE_Z, 
    'IMAGE_H'         : IMAGE_H,
    'IMAGE_W'         : IMAGE_W,
    'GRID_Z'          : GRID_Z,  
    'GRID_H'          : GRID_H,
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50,
}

def normalize(image):
    return image / 255.

train_imgs, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
### write parsed annotations to pickle for fast retrieval next time
#with open('train_imgs', 'wb') as fp:
#    pickle.dump(train_imgs, fp)

### read saved pickle of parsed annotations
#with open ('train_imgs', 'rb') as fp:
#    train_imgs = pickle.load(fp)
train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize, jitter=True,shuffle=True)

valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels=LABELS)
### write parsed annotations to pickle for fast retrieval next time
#with open('valid_imgs', 'wb') as fp:
#    pickle.dump(valid_imgs, fp)

### read saved pickle of parsed annotations
#with open ('valid_imgs', 'rb') as fp:
#    valid_imgs = pickle.load(fp)
valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False,shuffle=False)




#Setup a few callbacks and start the training

early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=100, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('weights_spine_3D.h5', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)


tb_counter  = len([log for log in os.listdir(os.path.expanduser('~/3dyolo/logs/')) if 'spine_' in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser('~/3dyolo//logs/') + 'spine_' + '_' + str(tb_counter), 
                          histogram_freq=0, 
                          write_graph=False, 
                          write_images=False)


optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)

model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = 200, 
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint, tensorboard ], 
                    max_queue_size   = 3)














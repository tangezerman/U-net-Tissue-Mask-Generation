#%% Import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

import numpy as np
import cv2

#%% Definitions
def TL_unet_model(input_shape):
    input_shape = input_shape
    base_VGG = VGG16(include_top = False, 
                   weights = "imagenet", 
                   input_shape = input_shape)


    for layer in base_VGG.layers: 
        layer.trainable = False


    bridge = base_VGG.get_layer("block5_conv3").output
    print(bridge.shape)


    layer_size = patch_size
    up1 = Conv2DTranspose(layer_size, (2, 2), strides=(2, 2), padding="same")(bridge)
    concat_1 = concatenate([up1, base_VGG.get_layer("block4_conv3").output], axis=3)
    conv6 = Conv2D(layer_size, (3, 3), activation="relu", padding="same")(concat_1)

    conv6 = Conv2D(layer_size, (3, 3), activation="relu", padding="same")(conv6)

    layer_size //= 2
    up2 = Conv2DTranspose(layer_size, (2, 2), strides=(2, 2), padding="same")(conv6)
    concat_2 = concatenate([up2, base_VGG.get_layer("block3_conv3").output], axis=3)
    conv7 = Conv2D(layer_size, (3, 3), activation="relu", padding="same")(concat_2)

    conv7 = Conv2D(layer_size, (3, 3), activation="relu", padding="same")(conv7)

    layer_size //= 2
    up3 = Conv2DTranspose(layer_size, (2, 2), strides=(2, 2), padding="same")(conv7)
    concat_3 = concatenate([up3, base_VGG.get_layer("block2_conv2").output], axis=3)
    conv8 = Conv2D(layer_size, (3, 3), activation="relu", padding="same")(concat_3)

    conv8 = Conv2D(layer_size, (3, 3), activation="relu", padding="same")(conv8)

    layer_size //= 2
    up4 = Conv2DTranspose(layer_size, (2, 2), strides=(2, 2), padding="same")(conv8)
    concat_4 = concatenate([up4, base_VGG.get_layer("block1_conv2").output], axis=3)
    conv9 = Conv2D(layer_size, (3, 3), activation="relu", padding="same")(concat_4)

    conv9 = Conv2D(layer_size, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)


    model_ = Model(inputs=[base_VGG.input], outputs=[conv10])

    return model_


class ImageMaskSequence(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, target_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
        self.indexes = np.arange(len(self.image_paths))

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in batch_indexes]
        batch_mask_paths = [self.mask_paths[k] for k in batch_indexes]

        images = np.array([self.load_image(path) for path in batch_image_paths])
        masks = np.array([self.load_mask(path) for path in batch_mask_paths])

        return images, masks

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = image / 255.0
        return image

    def load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)
        return mask

#%% Train

patch_size = 256
verison = 0
batch_size = 16

path = rf"Data\Data_{patch_size}_{verison}"
try:
    train_dir_images= os.path.join(path,"train","images")
    train_dir_masks = os.path.join(path,"train","masks")
     
    val_dir_images = os.path.join(path,"val","images") 
    val_dir_masks = os.path.join(path,"val","masks")  
except:
    print("Data file(s) missing")
    

target_size = (patch_size, patch_size)
input_shape = (patch_size, patch_size, 3) 


train_seq = ImageMaskSequence(train_dir_images, train_dir_masks, batch_size, target_size)
val_seq = ImageMaskSequence(val_dir_images, val_dir_masks, batch_size, target_size)


model = TL_unet_model(input_shape)

from tensorflow.keras import backend as K

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result
    
model.compile(optimizer="adam", 
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)])


callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss")]

history = model.fit(
train_seq,
epochs=10,
validation_data=val_seq,
callbacks = callbacks
)
print(model.metrics_names)
model_name = rf"model__{patch_size}_{batch_size}_{verison}"

model.save(os.path.join("Model",model_name))

# %% Test

def test_model(model, test_seq):

    evaluation_result = model.evaluate(test_seq)

    # Print the evaluation result
    print("Evaluation Result:", evaluation_result)
    
    return evaluation_result

test_dir_images = os.path.join(path, "test", "images")
test_dir_masks = os.path.join(path, "test", "masks")
test_seq = ImageMaskSequence(test_dir_images, test_dir_masks, batch_size, target_size)

test_model(model, test_seq)

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from data import * 
from unet import * 
from dense_unet import * 
from config import * 
from metrics import * 
from pretrained_backbone_unet import * 
import segmentation_models as sm
sm.set_framework('tf.keras')


#STANDARD UNET
#model = create_unet_model(SHAPE)

MODEL_ARCHITECTURE = sm.Unet('vgg19', encoder_weights='imagenet')

def plot_images(dataset):
    
    for x,y in dataset:
        print(x.shape)
        print(y.shape)

        print(x[0][125])
        print(y[0][125])


        plt.figure(figsize=(10, 10))
        img = x[0]

        plt.subplot(1, 2, 1)
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.imshow(y[0, :, :, 0])
        plt.show()


#images
image_dir = pathlib.Path(INPUT_PATH)
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]


mask_dir = pathlib.Path(GROUNDTRUTH_PATH)
mask_paths = list(mask_dir.glob('*.png'))
mask_paths = [str(path) for path in mask_paths]




x_train, x_val, y_train, y_val = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

num_train = len(x_train)
num_val = len(x_val)

print("Number of training examples: {}".format(num_train))
print("Number of validation examples: {}".format(num_val))


train_config = {
    'resize' : [SHAPE[0], SHAPE[0]],
    'scale'  : SCALE,
    'validate' : False
}
train_preprocess_fn = functools.partial(augment, **train_config)

validate_config = {
    'resize' : [SHAPE[0], SHAPE[0]],
    'scale'  : SCALE,
    'validate' : True
}
validate_preprocess_fn = functools.partial(augment, **validate_config)


# x_train = x_train[:1]
# y_train = y_train[:1]


train_dataset = create_dataset(x_train,y_train,preprocess_fn=train_preprocess_fn, batch_size=BATCH_SIZE)
validate_dataset = create_dataset(x_val,y_val,preprocess_fn=validate_preprocess_fn, batch_size=BATCH_SIZE)

#plot_images(train_dataset)


model = MODEL_ARCHITECTURE 


adam = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)


model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=[jaccard_loss, jaccard_index, dice_coeff, pixelwise_specificity, pixelwise_sensitivity, pixelwise_accuracy])

model.summary()


save_path = './models/attempt1.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_jaccard_index', save_best_only=True, verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_jaccard_index', min_delta=0, patience=4, verbose=0, mode='auto')

history = model.fit(train_dataset, 
                   steps_per_epoch=int(np.ceil(num_train / float(BATCH_SIZE))),
                   epochs=EPOCHS,
                   validation_data=validate_dataset,
                   validation_steps=int(np.ceil(num_val / float(BATCH_SIZE))),
                   callbacks=[checkpoint]
                   )

jaccard = history.history['jaccard_index']
val_jaccard = history.history['val_jaccard_index']

dice = history.history['dice_coeff']
val_dice = history.history['val_dice_coeff']


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs_range, jaccard, label='Training Jaccard Loss')
plt.plot(epochs_range, val_jaccard, label='Validation Jaccard Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Jaccard Index')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')


plt.subplot(2, 2, 3)
plt.plot(epochs_range, dice, label='Training Dice Coeff')
plt.plot(epochs_range, val_dice, label='Validation Dice Coeff')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.savefig('train-img/training_loss.png')


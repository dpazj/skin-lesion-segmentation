from __future__ import absolute_import, division, print_function, unicode_literals

import math
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




from sklearn.model_selection import train_test_split
from data import * 
from model import * 
from config import * 
from metrics import * 



#images
image_dir = pathlib.Path("../Data/ISIC2018/Training/ISIC2018_Task1-2_Training_Input")
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]

mask_dir = pathlib.Path("../Data/ISIC2018/Training/ISIC2018_Task1_Training_GroundTruth")
mask_paths = list(mask_dir.glob('*.png'))
mask_paths = [str(path) for path in mask_paths]

x_train, x_val, y_train, y_val = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

num_train = len(x_train)
num_val = len(x_val)

print("Number of training examples: {}".format(num_train))
print("Number of validation examples: {}".format(num_val))


train_config = {
    'resize' : [SHAPE[0], SHAPE[0]],
    'scale'  : SCALE
}
train_preprocess_fn = functools.partial(augment, **train_config)

validate_config = {
    'resize' : [SHAPE[0], SHAPE[0]],
    'scale'  : SCALE
}
validate_preprocess_fn = functools.partial(augment, **validate_config)


train_dataset = create_dataset(x_train,y_train,preprocess_fn=train_preprocess_fn, batch_size=BATCH_SIZE)
validate_dataset = create_dataset(x_val,y_val,preprocess_fn=validate_preprocess_fn, batch_size=BATCH_SIZE)




model = create_unet_model(SHAPE)

adam = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)

model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[dice_loss, jaccard_index])


model.summary()


save_path = './models/attempt1.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_dice_loss', save_best_only=True, verbose=1)

history = model.fit(train_dataset, 
                   steps_per_epoch=int(np.ceil(num_train / float(BATCH_SIZE))),
                   epochs=EPOCHS,
                   validation_data=validate_dataset,
                   validation_steps=int(np.ceil(num_val / float(BATCH_SIZE))),
                   callbacks=[checkpoint]
                   )

jaccard = history.history['jaccard_index']
val_jaccard = history.history['val_jaccard_index']

dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']


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
plt.plot(epochs_range, dice, label='Training Dice Loss')
plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.savefig('training_loss.png')
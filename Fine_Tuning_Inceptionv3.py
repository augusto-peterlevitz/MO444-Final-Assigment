from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras import backend as K
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
import glob
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.callbacks import LambdaCallback
import json
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_dir = '/content/drive/Assigment 05/chest_xray/train/AUGMENTED'
val_dir = '/content/drive/Assigment 05/chest_xray/val/AUGMENTED'
test_dir = '/content/drive/Assigment 05/chest_xray/test/'

#DATA PREPARATION/INCEPTIONV3 LOAD
n_classes = 2
IM_WIDTH = 200
IM_HEIGHT = 200
batch_size = 500
nb_epochs_first_train = 50
nb_epochs_second_train = 50

def n_samples(directory):
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
  train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0)
  val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0)
  train_generator = train_datagen.flow_from_directory(
          directory=train_dir,
          target_size=(IM_WIDTH, IM_HEIGHT),
          class_mode='categorical'
          )

  validation_generator = val_datagen.flow_from_directory(
          directory=val_dir,
          target_size=(IM_WIDTH, IM_HEIGHT),
          class_mode='categorical'
         	)
   
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

#PRE-TRAINING
nb_epoch = nb_epochs_first_train
number_train_samples = n_samples(train_dir)
number_val_samples = n_samples(val_dir)
validation_steps = number_val_samples/batch_size 
steps_per_epoch = number_train_samples/batch_size
json_log = open('/content/drive/Assigment 05/log_history.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'acc': logs['acc'], 'loss': logs['loss'], 'val_acc': logs['val_acc'], 'val_loss': logs['val_loss']})   + '\n'),
    on_train_end=lambda logs: json_log.close()
)

from keras.callbacks import ReduceLROnPlateau
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=2,
                                         min_lr=0.0000001,
                                         verbose=1)

filepath="/content/drive/Assigment 05/1st-TRAINING-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint, json_logging_callback, reduce_learning_rate]

history1=model.fit_generator(
        train_generator,
  	steps_per_epoch=steps_per_epoch,
  	nb_epoch=nb_epoch,
  	validation_data=validation_generator,
  	validation_steps=validation_steps, 
    callbacks=callbacks_list,
  	verbose = 1,
  	)

model_json = model.to_json()
with open("/content/drive/Assigment 05/Inception_Fine_Tuning_Model_pre-training.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/content/drive/Assigment 05/Inception_Fine_Tuning_Model_pre-training.h5")
print("Saved model to disk")
print('1st training finished')

#TRAINING/SECOND TRAIN
nb_epoch = nb_epochs_second_train
for layer in model.layers[:172]:
	layer.trainable = False

for layer in model.layers[172:]:
	layer.trainable = True

model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=2,
                                         min_lr=0.0000001,
                                         verbose=1)

json_log = open('/content/drive/Assigment 05/log_history_2ndTraining.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'acc': logs['acc'], 'loss': logs['loss'], 'val_acc': logs['val_acc'], 'val_loss': logs['val_loss']})   + '\n'),
    on_train_end=lambda logs: json_log.close()
)

filepath="/content/drive/Assigment 05/2ndTRAININGweights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint, json_logging_callback, reduce_learning_rate]

history=model.fit_generator(
        train_generator,
  	steps_per_epoch=steps_per_epoch,
  	nb_epoch=nb_epoch,
  	validation_data=validation_generator,
  	validation_steps=validation_steps,
    callbacks=callbacks_list,
  	verbose = 1,
  	class_weight=None)

print('2st training finished')
model_json = model.to_json()
with open("/content/drive/Assigment 05/Inception_Fine_Tuning_Model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/content/drive/Assigment 05/Inception_Fine_Tuning_Model.h5")
print("Saved model to disk")


#to test the model
test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0)
test_generator = test_datagen.flow_from_directory(
         directory=test_dir,
         target_size=(IM_WIDTH, IM_HEIGHT),
         class_mode='categorical')

steps = len(test_generator)
scores = model.evaluate_generator(test_generator, steps = steps, workers=1, use_multiprocessing=False, verbose=1)

Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
print(cm)
print('Classification Report')
target_names = ['NORMAL', 'PNEUMONIA']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    cmap=plt.cm.Blues
    cmap=plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')    
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
class_names = ["NORMAL", "PNEUMONIA"]
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Fine Tune Inception-V3 Normalized Confusion Matrix')

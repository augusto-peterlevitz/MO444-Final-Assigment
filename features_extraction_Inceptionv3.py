import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import os

train_path = r'\Users\Documents\assigment_05\chest_xray\train'
val_path  = r'\Users\Documents\assigment_05\chest_xray\val'
model_path = r'C:\Users\Documents\assigment_04\InceptionV3\tensorflow_inception_graph.pb'
test_path = r'\Users\Documents\assigment_05\chest_xray\test\graytoRGB_test'

def create_graph(model_path):
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(path, model_path):
    feature_dimension = 2048
    labels = []
    create_graph(model_path)
    i = 0
    
    samples_path =  os.path.join(path, 'NORMAL_AUGMENTED')
    samples_dir = os.listdir(samples_path)
    list_images = []
    count = 1
    
    for f in samples_dir:
        
        if ('jpeg' in f):
            list_images.append(f)
                  
    features = np.empty((len(list_images), feature_dimension))        
    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        
        for photo in list_images:
            photo_path = os.path.join(samples_path, photo)
            labels.append('NORMAL')            
            print('CLASS NORMAL: PROCESSING IMAGE %d OF %d)' % (count, len(list_images)))
            image_data = gfile.FastGFile(photo_path, 'rb').read()
            feature = sess.run(flattened_tensor, {'DecodeJpeg/contents:0': image_data})
            features[i, :] = np.squeeze(feature)
            i = i + 1
            count = count + 1
            
    features_normal = features

    samples_path =  os.path.join(path, 'PNEUMONIA_AUGMENTED')
    samples_dir = os.listdir(samples_path)
    list_images = []
    count = 1
    i = 0
    
    for f in samples_dir:
        if ('jpeg' in f):
            list_images.append(f)
            
    features = np.empty((len(list_images), feature_dimension))        
    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        
        for photo in list_images:
            photo_path = os.path.join(samples_path, photo)
            labels.append('PNEUMONIA')            
            print('CLASS PNEUMONIA: PROCESSING IMAGE %d OF %d)' % (count, len(list_images)))
            image_data = gfile.FastGFile(photo_path, 'rb').read()
            feature = sess.run(flattened_tensor, {'DecodeJpeg/contents:0': image_data})
            features[i, :] = np.squeeze(feature)
            i = i + 1
            count = count + 1

    features_pneumonia = features

    all_features = np.concatenate((features_normal, features_pneumonia), axis=0)        
    return all_features, labels

print('CALCULATING FEATURES OF THE TRAIN DATASET')     
(X_train, y_train) = extract_features(train_path, model_path)
np.savetxt('X_train.csv', X_train, delimiter=',')
y_train_array = np.array(y_train)
np.savetxt('y_train.csv', y_train_array, delimiter=',',fmt="%s")

print('CALCULATING FEATURES OF THE VALIDATION DATASET')
(X_val, y_val) = extract_features(val_path, model_path)
np.savetxt('X_val.csv', X_val, delimiter=',')
y_val_array = np.array(y_val)
np.savetxt('y_val.csv', y_val_array, delimiter=',',fmt="%s")

print('CALCULATING FEATURES OF THE TEST DATASET')
(X_test, y_test) = extract_features(test_path, model_path)
np.savetxt('X_test.csv', X_test, delimiter=',')
y_test_array = np.array(y_test)
np.savetxt('y_test.csv', y_test_array, delimiter=',',fmt="%s")


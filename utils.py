import numpy as np
import cv2
from os import listdir
from os.path import isdir
from numpy import asarray
import tensorflow as tf


def read_image(file):
  img = cv2.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def crop_bb(image, detection, margin):
    x1, y1, w, h = detection['box']
    x1 -= margin
    y1 -= margin
    w += 2*margin
    h += 2*margin
    if x1 < 0:
        w += x1
        x1 = 0
    if y1 < 0:
        h += y1
        y1 = 0
    return image[y1:y1+h, x1:x1+w]

def crop(mtcnn, img):
  det = mtcnn.detect_faces(img)
  ret = []
  boxes = []
  for i in det:
      margin = int(0.1 * img.shape[0])
      ret.append(crop_bb(img, i, margin))
      boxes.append(i['box'])
  return ret, boxes


def pre_process(mtcnn, face, required_size=(160, 160)):
    face, boxes = crop(mtcnn, face)
    rets = [cv2.resize(i, required_size) for i in face]
    re = []
    for ret in rets:
        ret = ret.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = ret.mean(), ret.std()
        ret = (ret - mean) / std
        re.append(ret)
    return re, boxes

# load images and extract faces for all images in a directory
def load_faces(mtcnn, directory):
    faces = list()
    # enumerate files
    i = 0
    for filename in listdir(directory):
        i += 1
        path = directory + filename
        img = read_image(path)
        face, _ = pre_process(mtcnn, img)
        face = face[0]
        faces.append(face)
        if i > 4:
            break
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(mtcnn, directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(mtcnn, path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

def load_tflite_model(file):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=file)
    interpreter.allocate_tensors()
    return interpreter

def predict(face_model, sample):
    # Get input and output tensors.
    input_details = face_model.get_input_details()
    output_details = face_model.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    outputs = []
    input_data = sample.reshape(input_shape)
    #input_data = np.expand_dims(input_data, axis=0)
    face_model.set_tensor(input_details[0]['index'], input_data)
    face_model.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = face_model.get_tensor(output_details[0]['index'])
    #print(output_data)
    outputs = output_data
    ret = np.stack(outputs)
    return ret[0]


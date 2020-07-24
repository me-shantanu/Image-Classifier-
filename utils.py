import tensorflow as tf
import tensorflow_hub as hub
import json


IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def get_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    # Class names contain index from 1 to 102, whereas the datasets have label indices from 0 to 101, hence     remapping
    class_names_new = dict()
    for key in class_names:
        class_names_new[str(int(key)-1)] = class_names[key]
    return class_names_new


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model

def process_image(numpy_image):
    print(numpy_image.shape)
    tensor_img = tf.image.convert_image_dtype(numpy_image, dtype=tf.int16, saturate=False)
    resized_img = tf.image.resize(numpy_image,(IMG_SIZE,IMG_SIZE)).numpy()
    norm_img = resized_img/255
    return norm_img
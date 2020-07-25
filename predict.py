import tensorflow as tf 
import tensorflow_hub as hub 
from PIL import Image
import json

import argparse
import numpy as np

# make all nessosory functions which we are going to use in our projects
def get_name(json_File):
    with open(json_File, 'r') as f:
        class_names =json.load(f)
    my_class = {}
    for i in class_names:
        my_class[str(int(i)-1)] = class_names[i]
    return my_class

def load(m_path):
    model = tf.keras.models.load_model(m_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model

def pre_process_image(np_image):
    tf_image = tf.image.convert_image_dtype(np_image, dtype=tf.int16, saturate=False)
    resize_imgage = tf.image.resize(tf_image,(30,30)).numpy()
    img = resize_imgage/255
    return img


def predict(image_path, model_path, top_k, all_class_names):
    top_k = int(top_k)
    model = load(model_path)

    img = Image.open(image_path)
    test_image = np.asarray(img)
    processed_test_image = pre_process_image(test_image)
    prob_preds = model.predict(np.expand_dims(processed_test_image,axis=0))
    prob_preds = prob_preds[0].tolist()
    top_pred_class_id = model.predict_classes(np.expand_dims(processed_test_image,axis=0))
    top_pred_class_prob = prob_preds[top_pred_class_id[0]]
    pred_class = all_class_names[str(top_pred_class_id[0])]
    print("\n\nMost likely class image and it's probability :\n","class_id :",top_pred_class_id, "class_name :", pred_class, "; class_probability :",top_pred_class_prob)
        

    values, indices= tf.math.top_k(prob_preds, k=top_k)

    probs_topk = values.numpy().tolist()#[0]
    classes_topk = indices.numpy().tolist()#[0]
    print("top k probs:",probs_topk)
    print("top k classes:",classes_topk)
    class_labels = [all_class_names[str(i)] for i in classes_topk]
    print('top k class labels:',class_labels)
    class_prob_dict = dict(zip(class_labels, probs_topk))       
    print("\nTop K classes along with associated probabilities :\n",class_prob_dict)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("image_path",help="Image Path", default="")
    parser.add_argument("saved_model",help="Model Path", default="")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 3)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()
    all_class_names = get_name(args.category_names)
    predict(args.image_path, args.saved_model, args.top_k, all_class_names)



import cv2
import tensorflow as tf
import tensorflow_hub as hub 
import json
import numpy as np

def categories(jsonfile):
    with open(jsonfile, 'r') as f:
        old_categories =json.load(f)
    new_categories = {}
    for i in old_categories:
        new_categories[str(int(i)-1)] = old_categories[i]
    return new_categories


def prepare(image_file_path):
    image_size = 224
    image_array = cv2.imread(image_file_path )
    new_array = cv2.resize(image_array,(image_size, image_size))
    return (new_array.reshape(-1, image_size,image_size,3))/255

my_model_path = "TrainedModel.h5"

model = tf.keras.models.load_model(my_model_path, custom_objects={'KerasLayer':hub.KerasLayer})


def predict_new_images(image_path):
    prediction = model.predict([prepare(image_path)])
    new_categories = categories("label_map.json")
    list_categories = list(new_categories.values())
    prob_preds = prediction[0].tolist()
    #i = prediction.index(max(prediction))
    
    top_pred_class_id = model.predict_classes(np.expand_dims([prepare(image_path)],axis=0))
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
  

predict_new_images("wild_pansy.jpg")
    
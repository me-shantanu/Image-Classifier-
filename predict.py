import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
from utils import process_image, get_class_names, load_model

import argparse
import numpy as np
import json

def predict(image_path, model_path, top_k, all_class_names):
    top_k = int(top_k)
    print(top_k, type(top_k))
    model = load_model(model_path)

    img = Image.open(image_path)
    test_image = np.asarray(img)

    # processing the image
    processed_test_image = process_image(test_image)
    
    # fetching prediction probabilities
    prob_preds = model.predict(np.expand_dims(processed_test_image,axis=0))
    prob_preds = prob_preds[0].tolist()
    
    
    #top 1 prediction
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

#     return probs_topk, classes_topk, top_pred_class_id, top_pred_class_prob


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("image_path",help="Image Path", default="")
    parser.add_argument("saved_model",help="Model Path", default="")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 3)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()

    all_class_names = get_class_names(args.category_names)
#     print("Displaying class names:\n",all_class_names)

    predict(args.image_path, args.saved_model, args.top_k, all_class_names)
#     probs, classes, top_class_id, top_class_prob = predict(args.image_path, args.saved_model, args.top_k, all_class_names)
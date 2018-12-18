import argparse
import matplotlib.pyplot as plt
import json
import os

"""
Utils library to handle input and output operations
@author: Steven Rojas <steven.rojas@gmail.com>
"""


def parse_arguments():
    """
    Parse input arguments for train script
    i.e.: python train.py --arch=vgg16 --learning_rate=0.01 --hidden_units="4096, 2048"
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path of flower images")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Path for save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Pre-trained Model")
    parser.add_argument("--learning_rate", type=float, default="0.01", help="Learning rate")
    parser.add_argument("--hidden_units", type=str, default="4096, 2048", help="Number of nodes per hidden layer")
    parser.add_argument("--epochs", type=int, default="10", help="Number of epochs")
    parser.add_argument("--gpu", help="Use GPU")

    return parser.parse_args()


def parse_predict_arguments():
    """
    Parse input arguments for predict script
    i.e.: python train.py image_path checkpoint
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path of flower image")
    parser.add_argument("checkpoint", type=str, help="Path of checkpoint")
    parser.add_argument("--top_k", type=str, default="5", help="Number of top classes")
    parser.add_argument("--category_names", default="cat_to_name.json", help="json file with flower names")
    parser.add_argument("--gpu", help="Use GPU")

    return parser.parse_args()


def get_flowers_names(cat_filename, flower_filename, classes, class_idx):
    file_path = os.getcwd() + "/" + cat_filename
    if not os.path.exists(file_path):
        print("Category file not found at {}".format(file_path))
        return False, False
    with open('cat_to_name.json', 'r') as f:
        names = json.load(f)
        idx_class = {val: key for key, val in class_idx.items()}
        labels = [idx_class[key] for key in classes]
        flowers = [names[key] for key in labels]
        code = flower_filename.split('/')[2]
        name = names[code]
        return name, flowers


def show_charts(values):
    plt.figure(figsize=(12, 8))
    t = plt.subplot(3, 3, 1)
    t.set_title("Training Loss")
    t.plot(values["training_loss_values"])
    v = plt.subplot(3, 3, 2)
    v.set_title("Validate Loss")
    v.plot(values["validate_loss_values"])
    a = plt.subplot(3, 3, 3)
    a.set_title("Accuracy")
    a.plot(values["accuracy_values"])
    plt.show()

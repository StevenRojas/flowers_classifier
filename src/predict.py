import os
import torch
from time import time
from utils import parse_predict_arguments, get_flowers_names
from predictor import Predictor

# predict.py floweres/test/10/image_07090.jpg checkpoints/checkpoint.pth


def main():
    args = parse_predict_arguments()
    print(args)  # TODO: Validate arguments and paths
    predictor = Predictor()
    load_checkpoint(predictor, args)
    device = get_device(args.gpu)
    load_model(predictor, device)
    image, probs, classes = predictor.predict(args.image_path, int(args.top_k))
    flower_name, flowers = get_flowers_names(args.category_names, args.image_path, classes,
                                             predictor.checkpoint['class_idx']['testing'])
    print("Flower name to predict: {}".format(flower_name))
    print("Prediction:")
    for name, p in zip(flowers, probs):
        print("\t -{}: {:.2f}%".format(name, p * 100))


def load_checkpoint(predictor, args):
    start_time = time()
    result = predictor.load_checkpoint(args)
    if result is False:
        print("[ERROR] Error while loading checkpoint: {}".format(predictor.last_error))
        exit(9)
    else:
        show_time(start_time, time(), "Checkpoint loaded")


def load_model(predictor, device):
    start_time = time()
    result = predictor.load_model(device)
    if result is False:
        print("[ERROR] Error while loading model: {}".format(predictor.last_error))
        exit(9)
    else:
        show_time(start_time, time(), "Model loaded and ready to use")


def get_device(gpu):
    device = "cpu"
    if gpu is not None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("Warning: CUDA is not available, using CPU")
    return device


def show_time(start, end, label):
    diff = end - start
    time_str = str(int((diff / 3600))) + ":" + str(int((diff % 3600) / 60)) + ":" + str(int((diff % 3600) % 60))
    print("{}, elapsed time: {}".format(label, time_str))


if __name__ == "__main__":
    main()
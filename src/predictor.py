import torch
import os

from arch_handler import ArchHandler
from images_handler import ImageHandler


class Predictor:
    """
    Class that load saved model and do predictions.
    @author: Steven Rojas <steven.rojas@gmail.com>
    """

    def __init__(self):
        self.last_error = None
        self.model = None
        self.architecture = None
        self.checkpoint = None
        self.device = None

    def load_checkpoint(self, args):
        checkpoint_path = os.getcwd() + "/" + args.checkpoint
        if not os.path.exists(checkpoint_path):
            self.last_error = "Checkpoint not found at {}".format(checkpoint_path)
            return False
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.checkpoint = checkpoint
        self.architecture = checkpoint['architecture']
        return True

    def load_model(self, device):
        try:
            self.device = device
            arch_handler = ArchHandler()
            print("Getting model...")
            self.model = arch_handler.load_model(self.architecture)
            self.model.load_state_dict(self.checkpoint["state_dict"])
            self.model.to(device)
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def predict(self, image_path, top_k):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        with torch.no_grad():
            image = ImageHandler().process_image(image_path)
            if self.device is "cuda":
                img = torch.from_numpy(image).type(torch.cuda.FloatTensor)
            else:
                img = torch.from_numpy(image).type(torch.FloatTensor)
            img = img.unsqueeze(0)
            output = self.model.forward(img)
            probs = torch.exp(output)
            top_probs, top_classes = probs.topk(top_k)
            top_probs = top_probs.cpu().data.numpy().squeeze()
            top_classes = top_classes.cpu().data.numpy().squeeze()
            return image, top_probs, top_classes

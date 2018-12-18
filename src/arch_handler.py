import torch
from torchvision import models
from network import Network


class ArchHandler:

    """
    Class to create architecture based on user parameters
    @author: Steven Rojas <steven.rojas@gmail.com>
    """

    def __init__(self):
        self.architecture = {
            "id": None,
            "input": None,
            "hidden": None,
            "output": 102,
            "h_act": Network.ACTIVATION_RELU,
            "o_act": Network.ACTIVATION_NONE,
            "drop_p": 0.3
        }

        self.supported_models = {
            "vgg16": self.__get_vgg16,
            "vgg13": self.__get_vgg13,
            "resnet18": self.__get_resnet18,
            "resnet101": self.__get_resnet101,
            "alexnet": self.__get_alexnet,
            "densenet121": self.__get_densenet121,
            "inception": self.__get_inception,
        }

    def create_model(self, args):
        hidden = [int(n) for n in args.hidden_units.split(',')]
        self.architecture["id"] = args.arch
        self.architecture["hidden"] = hidden
        model = self.__get_model(args.arch)
        return model

    def load_model(self, architecture):
        self.architecture["id"] = architecture['id']
        self.architecture["hidden"] = architecture['hidden']
        model = self.__get_model(architecture['id'])
        return model

    def __get_model(self, arch_name):
        func = self.supported_models.get(arch_name.lower(), lambda : "Not supported architecture")
        return func()

    def __get_vgg16(self):
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.architecture["input"] = model.classifier[0].in_features
         # TODO: Validate classifier architecture?
        model.classifier = Network(self.architecture)
        return model

    def __get_vgg13(self):
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.architecture["input"] = model.classifier[0].in_features
        model.classifier = Network(self.architecture)
        return model

    def __get_resnet18(self):
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.architecture["input"] = model.fc.in_features
        model.classifier = Network(self.architecture)
        return model

    def __get_resnet101(self):
        model = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.architecture["input"] = model.fc.in_features
        model.classifier = Network(self.architecture)
        return model

    def __get_alexnet(self):
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.architecture["input"] = model.classifier[1].in_features
        model.classifier = Network(self.architecture)
        return model

    def __get_densenet121(self):
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.architecture["input"] = model.classifier.in_features
        model.classifier = Network(self.architecture)
        return model

    def __get_inception(self):
        model = models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.architecture["input"] = model.fc.in_features
        model.classifier = Network(self.architecture)
        return model

import torch


class Trainer:
    """
    Class that generate different architectures, train and validate models.
    It saves the hyperparameters for each network configuration
    @author: Steven Rojas <steven.rojas@gmail.com>
    """

    def __init__(self, network, criterion, optimizer, device, scheduler=None):
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = None
        self.validloader = None
        self.testloader = None
        self.device = device
        self.epochs = None
        self.scheduler = scheduler

        self.training_loss_values = []
        self.validate_loss_values = []
        self.accuracy_values = []

    def set_train_loader(self, trainloader):
        self.trainloader = trainloader
        return self

    def set_valid_loader(self, validloader):
        self.validloader = validloader
        return self

    def set_test_loader(self, testloader):
        self.testloader = testloader
        return self

    def train(self, epochs, threshold=None):
        steps = 0
        running_loss = 0
        self.epochs = epochs
        for epoch in range(epochs):
            if self.scheduler is not None:
                self.scheduler.step()
            self.network.train()
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                steps += 1
                self.optimizer.zero_grad()
                output = self.network(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                self.__print_progress(steps)
                running_loss += loss.item()
            print("Epoch: {}/{}.. ".format(epoch + 1, epochs))
            accuracy = self.__validate_and_print(running_loss, epoch + 1)
            running_loss = 0
            steps = 0
            if threshold is not None and accuracy >= threshold:
                print("Stopping training due to reach threshold accuracy {:.3f}".format(accuracy))
                break

    def save_checkpoint(self, filename):
        checkpoint = {
            "architecture": self.network.classifier.get_architecture(),
            "state_dict": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion,
            "epochs": self.epochs,
            "class_idx": {
                "training": self.trainloader.dataset.class_to_idx,
                "validating": self.validloader.dataset.class_to_idx,
                "testing": self.testloader.dataset.class_to_idx,
            },
            "values": {
                "training_loss_values": self.training_loss_values,
                "validate_loss_values": self.validate_loss_values,
                "accuracy_values": self.accuracy_values,
            }
        }
        torch.save(checkpoint, filename)

    def __validate_model(self):
        accuracy = 0
        valid_loss = 0
        for inputs, labels in self.validloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output = self.network(inputs)
            valid_loss += self.criterion(output, labels).item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            eq = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(eq.type(torch.FloatTensor))

        return accuracy, valid_loss

    def __validate_and_print(self, running_loss, epoch):
        self.network.eval()
        with torch.no_grad():
            accuracy, valid_loss = self.__validate_model()

        length = len(self.validloader)
        tr_loss = running_loss / epoch
        t_loss = valid_loss / length
        ac = accuracy / length
        self.training_loss_values.append(tr_loss)
        self.validate_loss_values.append(t_loss)
        self.accuracy_values.append(ac.item())

        print("Training Loss: {:.3f}.. ".format(tr_loss),
              "Validate Loss: {:.3f}.. ".format(t_loss),
              "Accuracy: {:.3f}".format(ac))
        self.network.train()
        return ac

    def __print_progress(self, step):
        if step % 32 == 0:
            print('.')
        else:
            print('.', end='')

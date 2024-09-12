from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int, adapt_step: int, adapt_lr: float):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.adapt_step = adapt_step
        self.adapt_lr = adapt_lr

    @abstractmethod
    def train(self, dataset, net):
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def adapt_and_test(self, dataset, net):
        """
        Implement adapt and test method that evaluates the test_set of dataset on the given network.
        """
        pass

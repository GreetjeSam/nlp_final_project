from Training import Training


class Validation(Training):
    def __init__(self, epochs: list, learning_rates: list, optimizers: list) -> None:
        super().__init__()
        self.epochs = epochs
        self.learning_rates = learning_rates
        self.optimizers = optimizers
        
        # epochs: [200, 300]
        # learning rate: [0.0005, 0.001, 0.002]
        # optimizer: [Adam, Adadelta]
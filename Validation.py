from Encoder_Decoder import EngEncoder, NlDecoder
from Training import Training
from torch.utils.data import DataLoader
from torch import optim


class Validation(Training):
    def __init__(self, epochs: list, learning_rates: list, optimizers: list) -> None:
        super().__init__()
        self.epochs = epochs
        self.learning_rates = learning_rates
        self.optimizers = optimizers
        self.val_losses = []
        
    def run_validation(self, train_dataloader: DataLoader, encoder: EngEncoder, decoder: NlDecoder):
        for epoch_nr in self.epochs:
            for learning_rate in self.learning_rates:
                for optimizer in self.optimizers:
                    val_loss = self.train(train_dataloader, encoder, decoder, epoch_nr, optimizer, learning_rate)
                    self.val_losses.append([epoch_nr, learning_rate, str(optimizer), val_loss])

    def train(self, train_dataloader: DataLoader, encoder: EngEncoder, decoder: NlDecoder, n_epochs: int, 
              optimizer: optim.Adam | optim.Adadelta, learning_rate: float = 0.001, print_every: int = 2, plot_every: int = 2):
        return super().train(train_dataloader, encoder, decoder, n_epochs, optimizer, learning_rate, print_every, plot_every)
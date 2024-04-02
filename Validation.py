from Encoder_Decoder import EngEncoder, NlDecoder
from Training import Training
from torch.utils.data import DataLoader
from torch import optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Validation(Training):
    def __init__(self, epochs: list, learning_rates: list, optimizers: list) -> None:
        super().__init__()
        self.epochs = epochs
        self.learning_rates = learning_rates
        self.optimizers = optimizers
        self.val_losses = []
        
    def run_validation(self, val_dataloader: DataLoader, vocab_eng, vocab_nl, hidden_state_size, longest_sentence):
        index = 0
        for epoch_nr in self.epochs:
            encoder = EngEncoder(vocab_eng.num_words, hidden_state_size).to(device)
            decoder = NlDecoder(hidden_state_size, vocab_nl.num_words, vocab_nl, longest_sentence).to(device)
            for learning_rate in self.learning_rates:
                encoder = EngEncoder(vocab_eng.num_words, hidden_state_size).to(device)
                decoder = NlDecoder(hidden_state_size, vocab_nl.num_words, vocab_nl, longest_sentence).to(device)
                for optimizer in self.optimizers:
                    encoder = EngEncoder(vocab_eng.num_words, hidden_state_size).to(device)
                    decoder = NlDecoder(hidden_state_size, vocab_nl.num_words, vocab_nl, longest_sentence).to(device)
                    val_loss = self.train(val_dataloader, encoder, decoder, epoch_nr, optimizer, learning_rate, plot_name="validate"+str(index)+".png")
                    index += 1
                    self.val_losses.append([epoch_nr, optimizer, learning_rate, val_loss])
        return(self.choose_best())

    def choose_best(self):
        current_val_loss = 10000
        best_run = []
        for val_loss in self.val_losses:
            if val_loss[-1] < current_val_loss:
                current_val_loss = val_loss[-1]
                best_run = val_loss
        return best_run

    def train(self, train_dataloader: DataLoader, encoder: EngEncoder, decoder: NlDecoder, n_epochs: int, 
              optimizer: optim.Adam | optim.Adadelta, learning_rate: float = 0.001, plot_name = 'plot', print_every: int = 1, plot_every: int = 1):
        return super().train(train_dataloader, encoder, decoder, n_epochs, optimizer, learning_rate, plot_name, print_every, plot_every)


import math
import time
from Encoder_Decoder import EngEncoder, NlDecoder
from torch.utils.data import DataLoader
from torch.nn import Module as mod
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch

# The following source was used to calculate the validation loss: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Training():
    def __init__(self) -> None:
        pass

    def asMinutes(self, seconds: int):
        minutes = math.floor(seconds / 60)
        seconds -= minutes * 60
        return '%dm %ds' % (minutes, seconds)

    def timeSince(self, since: float, percent: float):
        now = time.time()
        seconds = now - since
        estimate = seconds / (percent)
        remaining = estimate - seconds
        return '%s (- %s)' % (self.asMinutes(seconds), self.asMinutes(remaining))
    
    def train_epoch(self, train_dataloader: DataLoader, encoder: EngEncoder, decoder: NlDecoder, encoder_optimizer, 
                    decoder_optimizer, criterion: CrossEntropyLoss):
        total_loss = 0

        for data in train_dataloader:
            input_tensor, target_tensor = data
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # resets gradients of all optimized tensors
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _, = decoder(encoder_outputs, encoder_hidden, target_tensor)

            # calculate the actual cross entropy loss with the decoder predictions and the targets
            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)),
                             target_tensor.view(-1))
            loss.backward()

            # perform one optimization step for both the encoder and decoder
            encoder_optimizer.step()
            decoder_optimizer.step()

            # the total loss over all the training data
            total_loss += loss.item()
        # return the total loss divided by the number of training samples
        return total_loss / len(train_dataloader)

    # hyper parameter tuning: nr epochs, learning rate, the adam optimizer
    def train(self, train_dataloader: DataLoader, validation_loader: DataLoader,encoder: EngEncoder, decoder: NlDecoder, n_epochs: int, 
              optimizer: (optim.Adam|optim.Adadelta), learning_rate: float =0.001, plot_name: str = 'plot',
               print_every: int =100, plot_every: int =100):
        start = time.time()
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        print_vloss = 0
        plot_vloss = 0
        plot_train_losses = []
        plot_val_losses = []

        encoder_optimizer = optimizer(mod.parameters(encoder), lr=learning_rate, weight_decay=1e-6)
        decoder_optimizer = optimizer(mod.parameters(decoder), lr=learning_rate, weight_decay=1e-6)
        
        # Using the cross entropy loss
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(train_dataloader, encoder, decoder, 
                               encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    input_tensor, target_tensor = vdata
                    input_tensor = input_tensor.to(device)
                    target_tensor = target_tensor.to(device)
                    encoder_outputs, encoder_hidden = encoder(input_tensor)
                    decoder_outputs, _, _, = decoder(encoder_outputs, encoder_hidden, target_tensor)
                    vloss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)),
                             target_tensor.view(-1))
                    print_vloss += vloss

            final_vloss = print_vloss / (i + 1)
            print_vloss = 0
            print("avg val loss", final_vloss.item())

            if epoch == n_epochs:
                final_loss = print_loss_total / print_every

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))
            
            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_vloss_avg = final_vloss
                plot_train_losses.append(plot_loss_avg)
                plot_val_losses.append(plot_vloss_avg.item())
                plot_loss_total = 0

        print(plot_train_losses, plot_val_losses)
        self.showPlot(n_epochs, plot_train_losses, plot_val_losses, plot_name)
        return final_loss
    
    def showPlot(self, epochs, train_loss: list, val_loss: list, plot_name):
        #plt.figure()
        epochs = range(1, epochs+1)
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.plot(epochs, train_loss, 'r', label='Training Loss')
        plt.title('Training and Validation Loss')        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(plot_name)

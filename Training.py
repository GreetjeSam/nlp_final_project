import math
import time
from torch.nn import Module as mod
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

class Training():
    def __init__(self) -> None:
        pass

    def asMinutes(self, seconds):
        minutes = math.floor(seconds / 60)
        seconds -= minutes * 60
        return '%dm %ds' % (minutes, seconds)

    def timeSince(self, since, percent):
        now = time.time()
        seconds = now - since
        estimate = seconds / (percent)
        remaining = estimate - seconds
        return '%s (- %s)' % (self.asMinutes(seconds), self.asMinutes(remaining))
    
    def showPlot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
    
    def train_epoch(self, train_dataloader, encoder, decoder, encoder_optimizer, 
                    decoder_optimizer, criterion):
        # this is just the training where the backpropegation is done
        # the validation does not do backpropegation, so we don't
        # add the validation to this. We do it after
        total_loss = 0

        for data in train_dataloader:
            input_tensor, target_tensor = data
            
            # resets gradients of all optimized tensors
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _, = decoder(encoder_outputs, encoder_hidden, target_tensor)

            #print(decoder_outputs.size(), target_tensor.size())

            # calculate the actual cross entropy loss with the decoder predictions and the targets
            #print(decoder_outputs.view(-1, decoder_outputs.size(-1)).size(), target_tensor.view(-1).size())
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
    def train(self, train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(mod.parameters(encoder), lr=learning_rate)
        decoder_optimizer = optim.Adam(mod.parameters(decoder), lr=learning_rate)
        
        # Using the cross entropy loss
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(train_dataloader, encoder, decoder, 
                               encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))
            
            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        self.showPlot(plot_losses)

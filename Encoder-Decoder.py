import torch
import torch.nn as nn
import torch.nn.functional as F
import MakeVocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EngEncoder(nn.Module):
    def __init__(self, eng_word_size, hidden_state_size, dropout_probability = 0.1):
        super(EngEncoder, self).__init__()
        self.hidden_size = hidden_state_size

        #using the embedding module to get word embeddings 
        self.embedding = nn.Embedding(eng_word_size,hidden_state_size)

        #using the GRU RNN to the embedded input with the hidden_state_size
        self.gru = nn.GRU(hidden_state_size,hidden_state_size,batch_first=True)

        #using the dropout to zero the elements in the input with the dropout_probability
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input):

        #embedding the input
        embedded = self.embedding(input)

        #zero some elements in the input
        dropout = self.dropout(embedded)

        #get the output tensor and the hiddenstate
        encoder_output, hidden_state = self.gru(dropout)
        return encoder_output, hidden_state
    

class Attention(nn.Module):
    def __init__(self, hidden_state_size):
        super(Attention, self).__init__()
        
        #apply the linear transformation
        self.q = nn.Linear(hidden_state_size, hidden_state_size)
        self.k = nn.Linear(hidden_state_size, hidden_state_size)
        self.v = nn.Linear(hidden_state_size, 1)

    def forward(self, query, keys):

        #caculate the scores of attention
        scores = self.v(torch.tanh(self.q(query) + self.k(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        #caculate the weights of attention
        weights = F.softmax(scores, dim=-1)

        #caculate the product of weights and keys
        prodcut_results = torch.bmm(weights, keys)

        return prodcut_results, weights

# not finished
class NlDecoder(nn.Module):
    def __init__(self, hidden_state_size, output_size, dropout_p=0.1):
        super(NlDecoder, self).__init__()

        #using the embedding module to get word embeddings 
        self.embedding = nn.Embedding(output_size, hidden_state_size)

        #get the attention by using attention class
        self.attention = Attention(hidden_state_size)


        self.gru = nn.GRU(2 * hidden_state_size, hidden_state_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_state_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(MakeVocab.index2word[1])
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MakeVocab.longest_sentence):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

    


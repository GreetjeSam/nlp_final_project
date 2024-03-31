import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EngEncoder(nn.Module):
    def __init__(self, eng_vocab_size: int, hidden_state_size: int, dropout_probability: float = 0.1):
        super(EngEncoder, self).__init__()
        self.hidden_size = hidden_state_size

        #using the embedding module to get word embeddings 
        self.embedding = nn.Embedding(eng_vocab_size,hidden_state_size)

        #using the GRU RNN to the embedded input which the size is the hidden_state_size
        self.encoderrnn = nn.GRU(hidden_state_size,hidden_state_size,batch_first=True)

        #using the dropout to zero the elements in the tensor with the dropout_probability
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input):

        #embedding the input
        embedded = self.embedding(input)

        #zero some elements in the input
        dropout = self.dropout(embedded)

        #get the output tensor and the hiddenstate
        encoder_output, hidden_state = self.encoderrnn(dropout)
        return encoder_output, hidden_state
    

class Attention(nn.Module):
    def __init__(self, hidden_state_size: int):
        super(Attention, self).__init__()
        
        #apply the linear transformation
        self.q = nn.Linear(hidden_state_size, hidden_state_size)
        self.k = nn.Linear(hidden_state_size, hidden_state_size)
        self.v = nn.Linear(hidden_state_size, 1)

    def forward(self, query, keys):

        #calculate the scores of every hidden state
        scores = self.v(torch.tanh(self.q(query) + self.k(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        #calculate the weights of attention
        weights = F.softmax(scores, dim=-1)

        #get the context vector 
        context_vector = torch.bmm(weights, keys)

        return context_vector, weights


class NlDecoder(nn.Module):
    def __init__(self, hidden_state_size: int, nl_vocab_size: int, vocab_nl, longest_sentence: int, dropout_probability: float =0.1):
        super(NlDecoder, self).__init__()

        #using the embedding module to get word embeddings 
        self.embedding = nn.Embedding(nl_vocab_size, hidden_state_size)

        #get the attention by using attention class
        self.attention = Attention(hidden_state_size)

        #using the GRU RNN to the embedded input which the size is double the hidden_state_size
        self.decoderrnn = nn.GRU(2 * hidden_state_size, hidden_state_size, batch_first=True)

        #using the output layer to get the output
        self.output_layer = nn.Linear(hidden_state_size, nl_vocab_size)

         #using the dropout to zero the elements in the tensor with the dropout_probability
        self.dropout = nn.Dropout(dropout_probability)
        self.vocab_nl = vocab_nl
        self.longest_sentence = longest_sentence

    def forward(self, encoder_outputs, encoder_hidden_state, target_tensor=None):

        #get the batch size 
        batch_size = encoder_outputs.size(0)
        # it also puts it to device here?
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(0) #0 is SOS_token

        #give the last hidden state from encoder to the decoder 
        decoder_hidden_state = encoder_hidden_state

        #create the list of decoder_ouput
        decoder_outputs = []

        #create the list of attetions
        attentions = []

        for i in range(self.longest_sentence):
            
            # get output, hidden_state, attn_weights in each step?
            decoder_output, decoder_hidden_state, attn_weights = self.forward_step(
                decoder_input, decoder_hidden_state, encoder_outputs
            )

            # add the decoder_output and attn_weights into their list
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        #concatenate the decoder_outputs in one dimension
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        #get the last-dimension tensor after using log_softmax
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        #concatenate the attention in one dimension
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden_state, attentions


    def forward_step(self, input, hidden, encoder_outputs):

        #embed the input and dropout 
        embedded =  self.embedding(input)
        dropout = self.dropout(embedded)

        query = hidden.permute(1, 0, 2)

        context, attn_weights = self.attention(query, encoder_outputs)

        #concatenate the dropouted and embedded input and the context vector in two dimensions
        input_gru = torch.cat((dropout, context), dim=2)

        output, hidden_state = self.decoderrnn(input_gru, hidden)

        #let the output go through the output layer
        output = self.output_layer(output)
        return output, hidden_state, attn_weights

    


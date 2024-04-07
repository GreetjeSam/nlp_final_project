import torch
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from FeatureExtraction import FeatureExtraction
from torch.utils.data import dataloader
from torchtext.data.metrics import bleu_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluation():
    def __init__(self, feat_extractor: FeatureExtraction, encoder, decoder, vocab_eng, vocab_nl) -> None:
        self.EOS_token = 1
        self.feat_extractor = feat_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_eng = vocab_eng
        self.vocab_nl = vocab_nl

    def evaluate(self, input_tensor):
        with torch.no_grad():
            #pass input through encoder decoder model
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = self.decoder(encoder_outputs, encoder_hidden)
            _, topi = torch.topk(decoder_outputs,1)
            decoded_ids = topi.squeeze()

            #convert the indices to words
            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == self.EOS_token:
                    decoded_words.append('EOS')
                    break
                decoded_words.append(self.vocab_nl.index2word[idx.item()])
        return decoded_words, decoder_attn
    
    def to_words(self, tensor):
        sentence = []
        tensor = tensor.tolist()
        for index in tensor[0]:
            if index != 0:
                sentence.append(str(self.vocab_nl.to_word(index)))
        sentence = [sentence]
        return sentence
    
    #evaluate the model on the test data using bleu score
    def evaluate_all_bleu(self, test_dataloader: dataloader):
        all_output_words = []
        references = []
        index = 0
        with torch.no_grad():
            for data_batch in test_dataloader:
                input_tensor, output_tensor = data_batch
                input_tensor = input_tensor.to(device)
                output_tensor = output_tensor.to(device)
                for tensor_in, tensor_target in zip(input_tensor, output_tensor):
                    references.append(self.to_words(tensor_target.view(1, -1)))
                    output_words, _ = self.evaluate(tensor_in.view(1, -1))
                    all_output_words.append(output_words)
                    if index < 10:
                        print("=", ' '.join(references[index][0]))
                        output_sentence = ' '.join(output_words)
                        print('<', output_sentence)
                    index += 1
        bleu = self.calc_bleu_score(all_output_words, references)
        return bleu
    
    def evaluateRandomly(self, paired_sent, n=10):
        for i in range(n):
            pair = random.choice(paired_sent)
            print('>', pair[0])
            print('=', pair[1])
            output_words, _ = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('') 

    def calc_bleu_score(self, candidate_corpus, references_corpus):
        return bleu_score(candidate_corpus, references_corpus)
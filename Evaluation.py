import torch
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from FeatureExtraction import FeatureExtraction
from torchtext.data.metrics import bleu_score

class Evaluation():
    def __init__(self) -> None:
        self.EOS_token = 1

    def evaluate(self, encoder, decoder, sentence, input_lang, output_lang):
        with torch.no_grad():
            feature_extractor = FeatureExtraction(input_lang.word2index, output_lang.word2index)
            input_tensor = feature_extractor.tensorFromSentence(input_lang.word2index, sentence)
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
            _, topi = torch.topk(decoder_outputs,2)
            decoded_ids = topi.squeeze()
            print(decoded_ids[:,1])

            decoded_words = []
            for idx in decoded_ids[:,1]:
                if idx.item() == self.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                decoded_words.append(output_lang.index2word[idx.item()])
        return(decoded_words, decoder_attn)
    
    def evaluateRandomly(self, encoder, decoder, paired_sent, n=10):
        for i in range(n):
            pair = random.choice(paired_sent)
            print('>', pair[0])
            print('=', pair[1])
            output_words, _ = self.evaluate(encoder, decoder, pair[0], self.input_lang, self.output_lang)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('') 

    def showAttention(self,input_sentence, output_words, attentions):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                        ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()

    def evaluateAndShowAttention(self, input_sentence):
        output_words, attentions = self.evaluate(self.encoder, self.decoder, input_sentence, self.input_lang, self.output_lang)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        self.showAttention(input_sentence, output_words, attentions[0, :len(output_words), :])


    def bleu_score(self, candidate_corpus, references_corpus):
        candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['No','Match']]
        references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
        return bleu_score(candidate_corpus, references_corpus)
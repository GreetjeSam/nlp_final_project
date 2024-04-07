import torch
from FeatureExtraction import FeatureExtraction
from torch.utils.data import dataloader
from torch.nn import CrossEntropyLoss
from torchtext.data.metrics import bleu_score

# The source for the evaluate method was inspired, but adapted, by: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# The rest of the class was a little inspired by the same source, but written ourselves.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluation():
    def __init__(self, feat_extractor: FeatureExtraction, encoder, decoder, vocab_eng, vocab_nl) -> None:
        self.EOS_token = 1
        self.feat_extractor = feat_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_eng = vocab_eng
        self.vocab_nl = vocab_nl
        self.reference_corpus = []
        self.test_loss = 0

    def evaluate(self, input_tensor, target_tensor, criterion):
        with torch.no_grad():
            #pass input through encoder decoder model
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = self.decoder(encoder_outputs, encoder_hidden)

            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)),
                             target_tensor.view(-1))
            self.test_loss += loss
            
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
        criterion = CrossEntropyLoss()
        with torch.no_grad():
            for data_batch in test_dataloader:
                input_tensor, output_tensor = data_batch
                input_tensor = input_tensor.to(device)
                output_tensor = output_tensor.to(device)
                for tensor_in, tensor_target in zip(input_tensor, output_tensor):
                    references.append(self.to_words(tensor_target.view(1, -1)))
                    output_words, _ = self.evaluate(tensor_in.view(1, -1), tensor_target, criterion)
                    all_output_words.append(output_words)
                    if index < 10:
                        print("=", ' '.join(references[index][0]))
                        output_sentence = ' '.join(output_words)
                        print('<', output_sentence)
                    index += 1
        self.reference_corpus = references
        bleu = self.calc_bleu_score(all_output_words)
        return bleu, float(self.test_loss / index)

    def calc_bleu_score(self, candidate_corpus):
        print(len(candidate_corpus), len(self.reference_corpus))
        return bleu_score(candidate_corpus, self.reference_corpus)

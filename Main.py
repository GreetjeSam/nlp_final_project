from Preprocessing import Preprocessing
from FeatureExtraction import FeatureExtraction
from MakeVocab import MakeVocab
from Training import Training
from Encoder_Decoder import EngEncoder, NlDecoder
from torch import optim
from Evaluation import Evaluation
import pickle
import torch
from Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    ############ preprocessing
    #run this code block once to create the cleaned_pairs.txt file, afterwards comment it out
    preprocesser = Preprocessing()
    # load English data
    filename_eng = 'europarl-v7.nl-en.en'
    eng_data = preprocesser.load_doc(filename_eng)
    # load Dutch data
    filename_nl = 'europarl-v7.nl-en.nl'
    nl_data = preprocesser.load_doc(filename_nl)

    # transform data into sentences and clean them
    sentences_eng = preprocesser.to_sentences(eng_data)
    sentences_eng = preprocesser.clean_lines(sentences_eng)

    # transform data into sentences and clean them
    sentences_nl = preprocesser.to_sentences(nl_data)
    sentences_nl = preprocesser.clean_lines(sentences_nl)

    paired_sent = []
    for eng_line, nl_line in zip(sentences_eng, sentences_nl):
        sen_nl = nl_line.split()
        sen_eng = eng_line.split()
        #remove sentences that are too long or too short
        if len(sen_nl) < 128 and len(sen_eng) < 128 and len(sen_nl) > 0 and len(sen_eng) > 0:
            paired_sent.append([eng_line, nl_line])

    preprocesser.save_clean_pairs(paired_sent, "cleaned_pairs.txt")

    '''
    ############ Load cleaned_pairs
    # load doc into memory
    with open("cleaned_pairs.txt", 'rb') as f:
        paired_sent = pickle.load(f)
        print("paired sentences loaded")
    # limit the number of sentences to your liking, to reduce training time
    paired_sent = paired_sent[:5000]
    
    ############ Feature extraction
    # run this to make new vocabularies, which will be saved in the current directory
    # comment this out after the vocabularies have been created
    vocab_eng_temp = MakeVocab()
    vocab_nl_temp = MakeVocab()
    vocab_eng_temp.make_vocab(paired_sent, 0)
    vocab_nl_temp.make_vocab(paired_sent, 1)
    print('made vocabulary')
    
    ############ model creation, training and evaluation
    #run this after first creating the vocabularies to load them
    vocab_eng = MakeVocab()
    vocab_nl = MakeVocab()
    vocab_eng.load_vocabularies(0)
    vocab_nl.load_vocabularies(1)
    print('Loaded vocabularies...')

    if vocab_eng.longest_sentence > vocab_nl.longest_sentence:
        longest_sentence = vocab_eng.longest_sentence
    else:
        longest_sentence = vocab_nl.longest_sentence

    feat_extraction = FeatureExtraction(vocab_eng.word2index, vocab_nl.word2index)
    train_dataloader, val_dataloader, test_dataloader = feat_extraction.get_dataloader(256, paired_sent, longest_sentence)

    hidden_state_size = 128
    encoder = EngEncoder(vocab_eng.num_words, hidden_state_size, longest_sentence+1).to(device)
    decoder = NlDecoder(hidden_state_size, vocab_nl.num_words, vocab_nl, longest_sentence+1).to(device)
    
    print("Training and validating...")
    trainer = Training()
    trainer.train(train_dataloader, val_dataloader, encoder, decoder, 15, optim.Adam, 0.001, plot_name="lossplot.png", print_every=1, plot_every=1)
    
    evaluator = Evaluation(feat_extraction, encoder, decoder, vocab_eng, vocab_nl)
    bleu, test_loss = evaluator.evaluate_all_bleu(test_dataloader)
    print("bleu: " + str(bleu))
    print("test loss: " + str(test_loss))
    '''
if __name__ == "__main__":
    main()
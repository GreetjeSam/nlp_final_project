from Preprocessing import Preprocessing
from FeatureExtraction import FeatureExtraction
from MakeVocab import MakeVocab
from Training import Training
import pickle


def main():
    
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
        paired_sent.append([eng_line, nl_line])

    preprocesser.save_clean_pairs(paired_sent, "cleaned_pairs.txt")
    
    # load doc into memory
    with open("cleaned_pairs.txt", 'rb') as f:
        paired_sent = pickle.load(f)

    paired_sent = paired_sent[:5000]

    vocab_eng = MakeVocab()
    vocab_nl = MakeVocab()
    vocab_eng.make_vocab(paired_sent, 0)
    vocab_nl.make_vocab(paired_sent, 1)

    if vocab_eng.longest_sentence > vocab_nl.longest_sentence:
        longest_sentence = vocab_eng.longest_sentence
    else:
        longest_sentence = vocab_nl.longest_sentence

    print(vocab_eng.to_word(3))
    print(vocab_nl.to_word(3))


    feat_extraction = FeatureExtraction(vocab_eng.word2index, vocab_nl.word2index)
    train_dataloader, val_dataloader, test_dataloader = feat_extraction.get_dataloader(40, paired_sent, longest_sentence)
    

if __name__ == "__main__":
    main()
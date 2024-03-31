from Preprocessing import Preprocessing
from FeatureExtraction import FeatureExtraction
from MakeVocab import MakeVocab
from Training import Training
from Encoder_Decoder import EngEncoder, NlDecoder
from Validation import Validation
from torch import optim
import pickle
import torch
from Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    '''preprocesser = Preprocessing()
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

        if len(sen_nl) < 128 and len(sen_eng) < 128 and len(sen_nl) > 0 and len(sen_eng) > 0:
            paired_sent.append([eng_line, nl_line])

    paired_sent = paired_sent[:5000]

    preprocesser.save_clean_pairs(paired_sent, "cleaned_pairs.txt")'''
    
    # load doc into memory
    with open("cleaned_pairs.txt", 'rb') as f:
        paired_sent = pickle.load(f)

    #paired_sent = paired_sent[:100]

    
    '''vocab_eng_temp = MakeVocab()
    vocab_nl_temp = MakeVocab()
    vocab_eng_temp.make_vocab(paired_sent, 0)
    vocab_nl_temp.make_vocab(paired_sent, 1)'''
    

    vocab_eng = MakeVocab()
    vocab_nl = MakeVocab()

    vocab_eng.load_vocabularies(0)
    vocab_nl.load_vocabularies(1)

    if vocab_eng.longest_sentence > vocab_nl.longest_sentence:
        longest_sentence = vocab_eng.longest_sentence
    else:
        longest_sentence = vocab_nl.longest_sentence

    #print(vocab_nl.num_words)
    #print(vocab_eng.to_word(3))
    #print(vocab_nl.to_word(3))
    #print(longest_sentence)

    feat_extraction = FeatureExtraction(vocab_eng.word2index, vocab_nl.word2index)
    train_dataloader, val_dataloader, test_dataloader = feat_extraction.get_dataloader(20, paired_sent, longest_sentence)
    
    hidden_state_size = 128
    encoder = EngEncoder(vocab_eng.num_words, hidden_state_size).to(device)
    decoder = NlDecoder(hidden_state_size, vocab_nl.num_words, vocab_nl, longest_sentence+1).to(device)
    
    '''validator = Validation(epochs=[2, 3], learning_rates=[0.001, 0.0025], optimizers=[optim.Adam, optim.Adadelta])
    best_paramters = validator.run_validation(train_dataloader, encoder, decoder)
    print(best_paramters)

    
    trainer = Training()
    trainer.train(train_dataloader, encoder, decoder, 10, optim.Adam ,print_every=2, plot_every=2)
    
    
    torch.save(encoder.state_dict(), "models\encoder_df1000_batch20_new.pt")
    torch.save(decoder.state_dict(), "models\decoder_df1000_batch20_new.pt")'''
    

    encoder.load_state_dict(torch.load("models/encoder_df1000_batch20_new.pt"))
    decoder.load_state_dict(torch.load("models/decoder_df1000_batch20_new.pt"))
    encoder.eval()
    decoder.eval()

    evaluator = Evaluation()
    evaluator.evaluate(encoder, decoder, "resumption now in session", vocab_eng, vocab_nl)

if __name__ == "__main__":
    main()
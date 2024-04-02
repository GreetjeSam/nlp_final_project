from Preprocessing import Preprocessing
from FeatureExtraction import FeatureExtraction
from MakeVocab import MakeVocab
from Training import Training
from Encoder_Decoder import EngEncoder, NlDecoder
from Validation import Validation
from torch import optim
from Evaluation import Evaluation
import pickle
import torch
from Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    '''
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

        if len(sen_nl) < 128 and len(sen_eng) < 128 and len(sen_nl) > 0 and len(sen_eng) > 0:
            paired_sent.append([eng_line, nl_line])

    preprocesser.save_clean_pairs(paired_sent, "cleaned_pairs.txt")
    '''
    # load doc into memory
    with open("cleaned_pairs.txt", 'rb') as f:
        paired_sent = pickle.load(f)
        print("paired sentences loaded")

    paired_sent = paired_sent[:200]
    '''
    vocab_eng_temp = MakeVocab()
    vocab_nl_temp = MakeVocab()
    vocab_eng_temp.make_vocab(paired_sent, 0)
    vocab_nl_temp.make_vocab(paired_sent, 1)
    print('made vocabulary')
    '''
    vocab_eng = MakeVocab()
    vocab_nl = MakeVocab()

    vocab_eng.load_vocabularies(0)
    vocab_nl.load_vocabularies(1)
    print('loaded vocabularies')

    if vocab_eng.longest_sentence > vocab_nl.longest_sentence:
        longest_sentence = vocab_eng.longest_sentence
    else:
        longest_sentence = vocab_nl.longest_sentence

    feat_extraction = FeatureExtraction(vocab_eng.word2index, vocab_nl.word2index)
    train_dataloader, val_dataloader, test_dataloader = feat_extraction.get_dataloader(200, paired_sent, longest_sentence)
    
    hidden_state_size = 256
    encoder = EngEncoder(vocab_eng.num_words, hidden_state_size).to(device)
    decoder = NlDecoder(hidden_state_size, vocab_nl.num_words, vocab_nl, longest_sentence+1).to(device)
    
    print("Validating on hyperparameters...")
    validator = Validation(epochs=[25], learning_rates=[0.0015, 0.0025, 0.003], optimizers=[optim.Adam])
    best_paramters = validator.run_validation(val_dataloader, vocab_eng, vocab_nl, hidden_state_size, longest_sentence+1)
    
    print("Training on best parameters...")
    trainer = Training()
    trainer.train(train_dataloader, encoder, decoder, best_paramters[0], best_paramters[1], best_paramters[2], print_every=5, plot_every=5)
    
    evaluator = Evaluation(feat_extraction, encoder, decoder, vocab_eng, vocab_nl)
    evaluator.evaluateRandomly(paired_sent)
    #print(evaluator.evaluate_all_bleu(paired_sent))


    '''
    torch.save(encoder.state_dict(), "nlp_final_project\models\encoder_df1000_batch64.pt")
    torch.save(decoder.state_dict(), "nlp_final_project\models\decoder_df1000_batch64.pt")
    '''
    '''
    encoder.load_state_dict(torch.load("models/encoder_df1000_batch20_new.pt"))
    decoder.load_state_dict(torch.load("models/decoder_df1000_batch20_new.pt"))
    encoder.eval()
    decoder.eval()
    '''

if __name__ == "__main__":
    main()
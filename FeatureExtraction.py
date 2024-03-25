import pickle
import MakeVocab
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class FeatureExtraction():
    def __init__(self, eng_vocab_word2index, nl_vocab_word2index) -> None:
        self.EOS_token = 1
        self.SOS_token = 0
        self.eng_vocab_word2index = eng_vocab_word2index
        self.nl_vocab_word2index = nl_vocab_word2index

    def indexesFromSentence(self, word2index, sentence):
        return [word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long).view(1, -1)

    def get_dataloader(self, batch_size, paired_sent, longest_sentence):

        n = len(paired_sent)
        input_ids = np.zeros((n, longest_sentence+1), dtype=np.int32)
        target_ids = np.zeros((n, longest_sentence+1), dtype=np.int32)
        
        for index, (input, target) in enumerate(paired_sent):
            inp_ids = self.indexesFromSentence(self.eng_vocab_word2index, input)
            tgt_ids = self.indexesFromSentence(self.nl_vocab_word2index, target)
            inp_ids.append(self.EOS_token)
            tgt_ids.append(self.EOS_token)
            input_ids[index, :len(inp_ids)] = inp_ids
            target_ids[index, :len(tgt_ids)] = tgt_ids

        print(input_ids[:5])
        print(target_ids[:5])
        

        all_data = TensorDataset(torch.LongTensor(input_ids),
                               torch.LongTensor(target_ids))
    
        train_size = int(0.49 * len(all_data))
        val_size = int(0.21 * len(all_data))
        test_size = len(all_data) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, val_size, test_size])


        #train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        return train_dataloader, val_dataloader, test_dataloader
        
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

# the following source was used to build this class: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# the train/validation/test split for the data loaders is inspired by the following source: https://www.d2l.ai/chapter_introduction/index.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtraction():
    def __init__(self, eng_vocab_word2index: dict, nl_vocab_word2index: dict) -> None:
        self.EOS_token = 1
        self.SOS_token = 0
        self.eng_vocab_word2index = eng_vocab_word2index
        self.nl_vocab_word2index = nl_vocab_word2index

    def indexesFromSentence(self, word2index: dict, sentence: str) -> list:
        return [word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang: dict, sentence: str) -> torch.Tensor:
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long).view(1, -1)

    def get_dataloader(self, batch_size: int, paired_sent: list, longest_sentence: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

        all_data = TensorDataset(torch.LongTensor(input_ids),torch.LongTensor(target_ids))
    
        train_size = int(0.64 * len(all_data))
        val_size = int(0.16 * len(all_data))
        test_size = len(all_data) - train_size - val_size
        
        #train, val, test split
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, val_size, test_size])

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        return train_dataloader, val_dataloader, test_dataloader
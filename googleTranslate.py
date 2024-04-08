from deep_translator import GoogleTranslator
from Evaluation import Evaluation
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Baseline():
    def __init__(self, evaluator: Evaluation) -> None:
        self.translator = GoogleTranslator(source="en", target="nl")
        self.translations = []
        self.evaluator = evaluator

    def to_words(self, tensor: torch.Tensor) -> list[list]:
        sentence = []
        tensor = tensor.tolist()
        for index in tensor[0]:
            if index != 0:
                sentence.append(str(self.evaluator.vocab_eng.to_word(index)))
        sentence = [sentence]
        return sentence

    def create_baseline(self, test_dataloader: DataLoader) -> list:
        for batch in test_dataloader:
            input_tensor, _ = batch
            input_tensor = input_tensor.to(device)
            for sentence in input_tensor:
                input_sentence = self.to_words(sentence.view(1, -1))
                for sent in input_sentence:
                    sent = ' '.join(sent)
                    translation = self.translator.translate(sent)
                    self.translations.append(translation.split())
        return self.translations
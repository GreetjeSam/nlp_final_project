class MakeVocab():
    def __init__(self) -> None:
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0
    
    def add_word(self, word):
            if word not in self.word2index:
                # First entry of word into vocabulary
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                # Word exists; increase word count
                self.word2count[word] += 1

    def add_sentence(self, sentence):
            sentence_len = 0
            for word in sentence.split(' '):
                sentence_len += 1
                self.add_word(word)
            if sentence_len > self.longest_sentence:
                # This is the longest sentence
                self.longest_sentence = sentence_len
            # Count the number of sentences
            self.num_sentences += 1

    def make_vocab(self, paired_sent, index):
         for pair in paired_sent:
              self.add_sentence(pair[index])

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]
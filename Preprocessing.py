import string
import re
import pickle
from unicodedata import normalize
# The following source was used for the clean_lines method: https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/

class Preprocessing():
    def __init__(self) -> None:
        pass

    def load_doc(self, filename: str):
        file = open(filename, mode='rt', encoding='utf-8')
        text = file.read()
        file.close()
        return text
    
    # split a loaded document into sentences
    def to_sentences(self, doc):
        return doc.strip().split('\n')

    # clean a list of lines
    def clean_lines(self, lines):
        cleaned = list()
        # prepare regex for char filtering
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for line in lines:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lower case
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            cleaned.append(' '.join(line))
        return cleaned

    def save_clean_pairs(self, paired_sent: list, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(paired_sent, f)
        print('Saved: %s' % filename)



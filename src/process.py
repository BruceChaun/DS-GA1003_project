import StringIO
from csv import reader
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import string
import re
import operator
import os
import random

random.seed(11111)

def spell_checker():
    # load word dictionary
    words = pickle.load(open('../glove/words.p', 'r'))

    path = ""
    raw_data_file = open(path + "elec_sub.csv", "rb")
    raw_data = reader(raw_data_file)
    header = next(raw_data) # skip header
    spells = []
    i = 1

    for row in raw_data:
        text = row[3].strip() + " " + row[2].strip()
        text = text.lower()
        text = split_sentences(text)
        text = filter(lambda w: w not in string.punctuation, text)
        text = filter(lambda w: not unicode(w, 'utf-8').isnumeric(), text)

        acc = 0
        for w in text:
            if w in words:
                acc += 1

        if len(text) > 0:
            rate = 1. - acc * 1. / len(text)
            spells.append(rate)
        else:
            spells.append(-1)

    pickle.dump(spells, open('spell.p', 'wb'))


def split_sentences(sent_seq):
    """
    sentence tokenization, mainly handles the case where a real period 
    followed by a new sentence without space.

    @param sent_seq: string, input sentence sequence
    @return word_token: a list of sentences
    """
    positions = findOccurencesOf(".!?", sent_seq)
    seq_len = len(sent_seq)
        
    if len(positions) > 0:
        for pos in list(reversed(positions)):
            if pos + 1 < seq_len:
                pos = pos + 1
                if sent_seq[pos] >= "A" and sent_seq[pos] <= "Z":
                    # add space if "." is followed by a capital letter
                    # I know this is a stupid idea
                    sent_seq = sent_seq[:pos] + " " + sent_seq[pos:]

    word_token = word_tokenize(sent_seq)
    return word_token


def findOccurencesOf(target, text):
    """
    find all the occurences of the substring in text

    @param target: string
    @param text: string
    @return positions: a list of positions where target starts
    """
    target = "[" + target + "]"
    positions = [m.start() for m in re.finditer(target, text)]
    return positions


def parse_numeric(string):
    """
    parse whether the whole string is a numeric, only including [0-9,.]

    For example,
        The followings are numeric
            1,000,000
            100000.00
            .11
            1,000.0

        While the followings are not numeric
            1,,000
            100.00.0
            10,00
            10000000,000
            1,000.

    @return True: if @string is numeric
            False: if @string is not numeric
    """
    string = string.strip()
    pattern = r"^(\d+(\.\d+)?|\d{1,3}(,\d\d\d)*(\.\d+)?|\.\d+)$"
    m = re.search(pattern, string)
    if m:
        return True
    return False


def read_words(filename, tau=0.5):
    """
    The text is a concatenation of summary and review text

    Note: 
        ignore all punctuations after tokenization, 
        parse numerics as special token <CD>

    @param filename: string, the input file
           tau: float, the threshold that label the helpfulness

    @return data: a list of tuple, which is composed of a token list and label
    """
    raw_data_file = open(filename, "rb")
    raw_data = reader(raw_data_file)
    header = next(raw_data) # skip header

    data = []
    for row in raw_data:
        try:
            helpful = int(row[11])
            total = int(row[12])
        except:
            continue
        y = 1 if helpful > total * tau else 0

        text = row[3].strip() + " " + row[2].strip()
        text = split_sentences(text)

        #filter all punctuations and replace numeric by <CD> -- numeric token
        text = filter(lambda w: w not in string.punctuation, text)
        text = ["<CD>" if parse_numeric(token) else token.lower() for token in text]

        data.append([text, y])

    return data


def get_vocab(filename="../data/vcb_20000.p"):
    """
    The vocabulary is a dict, whose key is word and value index
    """
    return pickle.load(open(filename, "rb"))


def word_count():
    """
    count the occurence of each token, sort the result descendantly
    """
    path = "../data/"
    filename = path + "texts.p"
    idx_words = pickle.load(open(filename, "rb"))

    wc = {}
    for tokens in idx_words:
        for token in tokens:
            wc[token] = wc.get(token, 0) + 1

    print len(wc)
    wc = sorted(wc.items(), key=operator.itemgetter(1), reverse=True)
    pickle.dump(wc, open(path+"wc.p", "wb"))


def top_frequent_tokens(n):
    """
    top @n frequent tokens are regarded as vocabulary, the words outside 
    the vocabulary are marked as "<UNK>"

    @param n: int, specifying the vocabulary size
    @return vcb: dict, where key is *token*, value is its *index*
    """
    path = "../data/"
    filename = path + "wc.p"
    wc = pickle.load(open(filename, "rb")) # the word size has more than 680K
    wc = wc[:n]

    vcb = {}
    for pair in wc:
        k = pair[0]
        vcb[k] = len(vcb)
    vcb["<UNK>"] = len(vcb) # add <UNK> token
    
    name = "vcb_%d.p" % n
    pickle.dump(vcb, open(path+name, "wb"))
    return vcb


def file_to_word_ids(filename, vocab):
    """
    Convert words in file into index

    @param filename: string, a file that contains review texts
           vocab: dict, set those words outside of vocabulary as "<UNK>"
    @return idx_words: a list of a list of integer representing each token
    """
    idx_words = read_words(filename, 0.7)

    for i in range(len(idx_words)):
        for j in range(len(idx_words[i][0])):
            if idx_words[i][0][j] not in vocab:
                idx_words[i][0][j] = vocab["<UNK>"]
            else:
                idx_words[i][0][j] = vocab[idx_words[i][0][j]]

    return idx_words


def split_data(path="../data", filename="elec_sub.csv"):
    """
    split the whole dataset into train, valid and test set
    """
    with open(os.path.join(path, filename), "rb") as f:
        data = f.read().split('\n')
    random.shuffle(data)

    n = len(data)
    n_train = int(n * 0.6)
    n_valid = int(n * 0.8)

    train_data = data[0:n_train]
    valid_data = data[n_train:n_valid]
    test_data = data[n_valid:]

    with open(os.path.join(path, "train.csv"), "w") as f:
        f.write("\n".join(train_data))

    with open(os.path.join(path, "valid.csv"), "w") as f:
        f.write("\n".join(valid_data))

    with open(os.path.join(path, "test.csv"), "w") as f:
        f.write("\n".join(test_data))


def padding(data, pad, max_len=None):
    """
    padding the data such that each sequence has the same length

    len(vocab) as padding value
    """
    if not max_len:
        seq_len = [len(x) for x in data]
        max_len = max(seq_len)

    for i in range(len(data)):
        data[i] += [pad] * (max_len-len(data[i]))

    return data


def get_raw_data(path):
    """
    @param path: string, a dir that contains training, validation, and test data
    @return train, valid and test data
    """
    train_path = os.path.join(path, "train.csv")
    valid_path = os.path.join(path, "valid.csv")
    test_path = os.path.join(path, "test.csv")

    vcb = get_vocab()
    train_data = file_to_word_ids(train_path, vcb)
    valid_data = file_to_word_ids(valid_path, vcb)
    test_data = file_to_word_ids(test_path, vcb)

    return train_data, valid_data, test_data, len(vcb)


def load_raw_data(filename):
    data = pickle.load(open(filename, "rb"))
    return data


def generate_data():
    n = 20000
    top_frequent_tokens(n)
    train_data, valid_data, test_data, vcb_sz = get_raw_data("../data")
    pickle.dump(train_data, open("../data/train.p", "wb"))
    pickle.dump(valid_data, open("../data/valid.p", "wb"))
    pickle.dump(test_data, open("../data/test.p", "wb"))


def truncate(filename):
    vocab = get_vocab()
    raw_data_file = open(filename, "rb")
    out = open(filename+".trunc", "w")
    raw_data = reader(raw_data_file)
    tau = 0.7
    data = []

    for row in raw_data:
        if len(row) == 0:
            continue
        text = row[3].strip() + " " + row[2].strip()
        text = split_sentences(text)
        if len(text) > 0 and len(text) < 500:
            row[2] = '"' + row[2] + '"'
            row[3] = '"' + row[3] + '"'
            out.write(', '.join(row) + '\n')

            try:
                helpful = int(row[11])
                total = int(row[12])
            except:
                continue
            y = 1 if helpful > total * tau else 0

            text = row[3].strip() + " " + row[2].strip()
            text = split_sentences(text)

            #filter all punctuations and replace numeric by <CD> -- numeric token
            text = filter(lambda w: w not in string.punctuation, text)
            text = ["<CD>" if parse_numeric(token) else token.lower() for token in text]

            data.append([text, y])

    idx_words = data
    for i in range(len(idx_words)):
        for j in range(len(idx_words[i][0])):
            if idx_words[i][0][j] not in vocab:
                idx_words[i][0][j] = vocab["<UNK>"]
            else:
                idx_words[i][0][j] = vocab[idx_words[i][0][j]]
    
    print(len(idx_words))
    pickle.dump(idx_words, open("../data/test500.p", "wb"))



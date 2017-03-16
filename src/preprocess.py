from csv import reader
import StringIO
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize

np.random.seed(1003)

def process_raw_data():
    """
    pre processing the raw data.

    Tokenize sentences and words to count to number.
    """
    print "processing raw data ... "

    path = "../data/"
    raw_data_file = open(path + "elec_sub.csv", "rb")
    processed = open(path + "elec_sub_processed.csv", "w")
    delimit = ","

    raw_data = reader(raw_data_file)
    header = next(raw_data) # skip header
    header.append("#sentence")
    header.append("#word_token")

    strip_list(header)
    processed.write(delimit.join(header) + "\n")

    i = 1
    for row in raw_data:
        strip_list(row)
        sentences = split_sentences(row[2])
        word_token = []
        for sentence in sentences:
            word_token += word_tokenize(sentence)
        row[2] = '"' + " ".join(word_token) + '"'
        row[3] = '"' + row[3] + '"'
        row[13] = '"' + row[13] + '"'
        row.append(str(len(sentences)))
        row.append(str(len(word_token)))

        processed.write(delimit.join(row) + "\n")

    processed.close()

def strip_list(row):
    """
    Given a list of strings, strip spaces for each element

    @param row: a list of strings
    """
    for i in range(len(row)):
        row[i] = row[i].strip()

def split_sentences(sent_seq):
    """
    sentence tokenization, mainly handles the case where a real period 
    followed by a new sentence without space.

    @param sent_seq: string, input sentence sequence
    @return sent_token: a list of sentences
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

    sent_token = sent_tokenize(sent_seq)
    return sent_token

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

def load_data(path):
    f = open(path, "r")
    data = []
    line = f.readline() # skip header
    line = f.readline()
    noise = 0

    while line:
        row = next(reader(StringIO.StringIO(line)))

        if len(row) != 28:
            noise += 1
        else:
            data.append(line)

        line = f.readline()

    print noise, len(data)
    return data

def split_data():
    """
    split preprocessed data into training, validation and test set, 
    with proportion 60%, 20%, 20%
    """
    print "spliting raw data into three sets ... "
    path = "../data/"
    data_path = path + "elec_sub_processed.csv"
    data = load_data(data_path)

    num_data = len(data)
    num_train = int(num_data * 0.6)
    num_valid = int(num_data * 0.2)

    # shuffle the raw data
    order = list(range(num_data))
    np.random.shuffle(order)

    # generate train data
    train_data = open(path + "train.csv", "w")
    for i in order[:num_train]:
        train_data.write(data[i])
    train_data.close()

    # generate validation set
    valid_data = open(path + "valid.csv", "w")
    for i in order[num_train:num_train+num_valid]:
        valid_data.write(data[i])
    valid_data.close()

    # generate validation set
    test_data = open(path + "test.csv", "w")
    for i in order[num_train+num_valid:]:
        test_data.write(data[i])
    test_data.close()


if __name__ == "__main__":
    process_raw_data()
    split_data()

from csv import reader
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize

def process_raw_data():
    """
    pre processing the raw data.

    Tokenize sentences and words to count to number.
    """
    print "processing raw data ... "

    path = "../data/"
    raw_data_file = open(path + "elec_sub.csv", "rb")
    processed = open(path + "elec_sub_processed.csv", "w")
    delimit = ", "

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

if __name__ == "__main__":
    process_raw_data()

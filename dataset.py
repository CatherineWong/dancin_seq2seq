"""
dataset.py

Classes:
    DatasetEncoderDecoder: encodes and decodes sentences according to a fixed, written vocabulary.
    SpamDataset: utility functions to read and write dataset files.
"""

import os
import numpy as np
import sklearn

class DatasetEncoderDecoder(object):
    """
    Encodes and decodes sentences according to a vocabulary.
    
    Sentences are truncated. OOV words are assigned an <UNK> token, and <SOS>, <PAD>, and <EOS> tokens are added.
    
    truncation_len
    """
    def __init__(self, vocab_file, truncation_len=100):
        self.truncation_len = truncation_len
        # Create index to word and word to index dicts from the vocab_file.
        num_default_tokens = 4
        self.index2word = {0:'<SOS>', 1:'<EOS>', 2: '<UNK>', 3: '<PAD>'}
        self.word2index = {'<SOS>':0, '<EOS>':1, '<UNK>': 2, '<PAD>': 3}
        with open(vocab_file) as f:
            all_lines = [line.strip() for line in f.readlines()]
        for idx, token in enumerate(all_lines):
            self.index2word[idx + num_default_tokens] = token
            self.word2index[token] = idx + num_default_tokens
          
    def encode(self, sentence):
        """
        Encodes a sentence according to the vocabulary.
        Returns:
            normalized: the normalized sentence, as it would be decoded.
            encoded: the space-separated numerical sentence.
        """
        truncated = sentence.lower().split()[:self.truncation_len]
        truncated += ['<PAD>'] * max(self.truncation_len - len(truncated), 0)
        truncated = ['<SOS>'] + truncated + ['<EOS>']
        
        normalized = []
        encoded = []
        # Encode, removing the UNK tokens
        for token in truncated:
            token = token if token in self.word2index else '<UNK>'
            normalized.append(token)
            encoded.append(str(self.word2index[token]))
        
        normalized = " ".join(normalized)
        encoded = " ".join(encoded)
        return normalized, encoded
    
    def decode_numpy(self, numerical_encoded):
        """Returns the decoded sentence."""
        return " ".join([self.index2word[token] for token in numerical_encoded])

    def decode(self, encoded):
        """Returns the decoded sentence."""
        numerical_encoded = [int(token) for token in encoded.split()]
        return " ".join([self.index2word[token] for token in numerical_encoded])

    
class SpamDataset(object):
    """
    Dataset: encapsulates utility functions to get the dataset files.
    """
    def __init__(self,
                 base_data_dir="/cvgl2/u/catwong/cs332_final_project/data/",
                 splits=['train', 'val', 'test'],
                 label_names=['ham', 'spam'],
                 truncation_len=100,
                 encoded_files=['encoded_ham.txt', 'encoded_spam.txt'],
                 vocab_file='email_train_vocab.txt',
                 random_seed=10):
        self.base_data_dir = base_data_dir
        self.splits = splits
        self.label_names = label_names
        self.encoded_files = [str(truncation_len) + "_" + f for f in encoded_files]
        self.vocab_file = os.path.join(base_data_dir, str(truncation_len) + "_" + vocab_file)
        self.vocab_encoder = DatasetEncoderDecoder(self.vocab_file, truncation_len=truncation_len)
        self.random_seed = random_seed
        
        # Read in all of the lines from the files.
        self.examples_dict = {}
        self.labels_dict = {}
        for split in splits:
            all_examples = []
            all_labels = []
            for label, encoded_file in enumerate(self.encoded_files):
                data_file = os.path.join(base_data_dir, split, encoded_file)
                with open(data_file) as f:
                    all_lines = [line.strip().split() for line in f.readlines()]
                all_examples += all_lines
                all_labels += [label] * len(all_lines)
            self.examples_dict[split] = all_examples
            self.labels_dict[split] = all_labels
            
    
    def examples(self, 
                 split, 
                 shuffled=False):
        """
        Args:
            split: one of the splits (ex. train, val, test) with labels.
            shuffled: whether to shuffle the examples.(default: True)
        Returns:
            examples: (list of lists)
            labels: (list)
        """
        examples = np.array(self.examples_dict[split]).astype(int)
        labels = np.array(self.labels_dict[split])
        if shuffled:
            examples, labels = sklearn.utils.shuffle(examples, labels, random_state=self.random_seed)
        return examples, labels
    
    def dataset_stats(self):
        """Prints useful stats about the dataset."""
        for split in self.splits:
            labels = self.labels_dict[split]
            num_pos = np.sum(labels)
            num_neg = len(labels) - num_pos
            print "Total %s examples: %d, %s: %d, %s: %d" % (split, 
                                                             len(labels),
                                                             self.label_names[0], 
                                                             num_neg, 
                                                             self.label_names[1], 
                                                             num_pos)
if __name__ == "__main__":
    # SpamDataset demonstration.
    print "SpamDataset demo:"
    for truncation_len in [30, 100]:
        dataset = SpamDataset(truncation_len=truncation_len)
        examples, labels =  dataset.examples(split='train', shuffled=True)
        print examples[0]
        print labels[0]
        print dataset.vocab_encoder.decode(" ".join(examples[0].astype(str)))
        dataset.dataset_stats()
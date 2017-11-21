"""
autoencoder.py - Autoencoder classes.

Classes:
    Autoencoder: a general autoencoder interface.
    SpamSeq2SeqAutoencoder: a sequence to sequence autoencoder interface.

"""

from __future__ import division

import logging
import numpy as np
import os
import scipy
import scipy.stats
import sklearn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import SpamDataset
from seq2seq.model import Seq2Seq, Seq2SeqAutoencoder

use_cuda = torch.cuda.is_available()
print "Use CUDA:" + str(use_cuda)

class Autoencoder(object):
    """
    Autoencoder: a general discriminator class.
    """
    def __init__(self, checkpoint=None, dataset=None):
        pass
    
    def train(self, dataset):
        raise Exception("Not implemented")
        
    def evaluate(self, dataset, split, verbose=True):
        raise Exception("Not implemented")
    
    def save_model(self):
        # Outputs a path that can be passed into the restore.
        raise Exception("Not implemented")
    
    def restore_model(self, model_checkpoint):
        raise Exception("Not implemented")
        
class SpamSeq2SeqAutoencoder(Autoencoder):
    """
    SpamSeq2Seq Autoencoder.
    Implementation from: https://github.com/MaximumEntropy/Seq2Seq-PyTorch
    Uses the following config: config_en_autoencoder_1_billion.json
    """
    def __init__(self, truncation_len=100, checkpoint=None, dataset=None):
        Autoencoder.__init__(self, checkpoint, dataset)
        
        self.truncation_len = truncation_len
        self.dataset = dataset 
        self.vocab_size = len(self.dataset.vocab_encoder.word2index)
        self.pad_token_ind = self.dataset.vocab_encoder.word2index['<PAD>']
        self.batch_size = 50
        
        # Initialize the model.
        self.model = Seq2SeqAutoencoder(
            src_emb_dim=256,
            trg_emb_dim=256,
            src_vocab_size=self.vocab_size,
            src_hidden_dim=512,
            trg_hidden_dim=512,
            batch_size=self.batch_size,
            bidirectional=True,
            pad_token_src=self.pad_token_ind,
            nlayers=2,
            nlayers_trg=1,
            dropout=0.,
        ).cuda()
        
        # Restore from checkpoint if provided.
        if checkpoint:
            self.restore_model(checkpoint)
        
        # Initialize the optimizer.
        self.lr = 0.0002
        self.clip_c = 1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Loss criterion.
        weight_mask = torch.ones(self.vocab_size).cuda()
        weight_mask[self.pad_token_ind] = 0
        self.loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
        
        # Save the initial model.
        self.save_model()
        
    def clip_gradient(self, model, clip):
        """Compute a gradient clipping coefficient based on gradient norm."""
        totalnorm = 0
        for p in self.model.parameters():
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
        totalnorm = math.sqrt(totalnorm)
        return min(1, clip / (totalnorm + 1e-6))
    
    def get_dataset_minibatch(self, examples, iter_ind, batch_size):
        """
        Iterator over the dataset split and get autoencoder minibatches.
        """
        minibatch = examples[iter_ind:iter_ind+batch_size]
        
        # Create the Pytorch variables.
        input_lines = Variable(torch.LongTensor(np.fliplr(minibatch).copy())).cuda() # Reverse the input lines.
        output_lines = Variable(torch.LongTensor(minibatch)).cuda()
        return input_lines, output_lines
    
    def perplexity(self):
        """Calculate the BLEU score."""
        
    def train(self, dataset, epochs=2, write_checkpoint=1, monitor_loss=1, print_samples=1):
        examples, _ = dataset.examples(split="train", shuffled=True)
        num_examples, max_len = examples.shape
        
        for epoch in xrange(epochs):
            losses = []
            for iter_ind in xrange(0, num_examples, self.batch_size):
                # Get a minibatch.
                input_lines_src, output_lines_src = self.get_dataset_minibatch(examples, iter_ind, self.batch_size)
                
                # Run a training step.
                decoder_logit = self.model(input_lines_src)
                self.optimizer.zero_grad()

                loss = self.loss_criterion(
                    decoder_logit.contiguous().view(-1, self.vocab_size),
                    output_lines_src.view(-1)
                )
                losses.append(loss.data[0])
                loss.backward()
                self.optimizer.step()
                
                if iter_ind % monitor_loss == 0:
                    logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (epoch, iter_ind, np.mean(losses)))
                    del losses
                    losses = []
                
                if iter_ind % print_samples == 0:
                    # Print samples.
                    word_probs = self.model.decode(decoder_logit).data.cpu().numpy().argmax(axis=-1)
                    output_lines_trg = input_lines_src.data.cpu().numpy()
                    for sentence_pred, sentence_real in zip(word_probs[:5], output_lines_trg[:5]):
                        decoded_real = dataset.vocab_encoder.decode_numpy(sentence_real[::-1])
                        decoded_pred = dataset.vocab_encoder.decode_numpy(sentence_pred)
                        
                        logging.info('===============================================')
                        logging.info("REAL: " + decoded_real)
                        logging.info("PREDICTED: " + decoded_pred)
                        logging.info('===============================================')
            if epoch % write_checkpoint == 0:
                self.save_model()
            
        
    def evaluate(self, dataset, split, verbose=True):
        raise Exception("Not implemented")
    
    def save_model(self, 
                   checkpoint_dir='/cvgl2/u/catwong/cs332_final_project/checkpoints',
                   checkpoint_name='seq2seq_autoencoder'):
        # Outputs a path that can be passed into the restore.
        checkpoint_file = str(self.truncation_len) + "_" + checkpoint_name + '.model'
        full_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        torch.save(
            self.model.state_dict(),
            open(full_checkpoint_path, 'wb')
        )
        return full_checkpoint_path
    
    def restore_model(self, checkpoint):
        self.model.load_state_dict(torch.load(open(checkpoint)))

# Demo
if __name__ == "__main__":
    truncation_len=100
    experiment_name = str(truncation_len) + "_autoencoder"
    
    # Initialize logging.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='log/%s' % (experiment_name),
        filemode='w'
    )
    
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    # optional, set the logging leveld
    console.setLevel(logging.INFO)
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    
    # Train the autoencoder.
    spam_dataset = SpamDataset(truncation_len=truncation_len)
    autoencoder = SpamSeq2SeqAutoencoder(truncation_len=truncation_len, dataset=spam_dataset)
    autoencoder.train(
        dataset=spam_dataset, 
        epochs=1000, 
        write_checkpoint=1,
        monitor_loss=5000,
        print_samples=10000)
    checkpoint = autoencoder.save_model()
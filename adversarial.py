"""
adversarial.py - Adversarial classes.

Classes:
    Autoencoder: a general autoencoder interface.
    SpamSeq2SeqAutoencoder: a sequence to sequence autoencoder interface.
"""

from __future__ import division

import gc
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
from discriminator import MultinomialNBDiscriminator
from seq2seq.model import Seq2Seq, Seq2SeqAutoencoder

use_cuda = torch.cuda.is_available()
print "Use CUDA:" + str(use_cuda)

class DancinSeq2SeqAdversarial():
    """
    Dancin Seq2Seq Adversarial Text Generation..
    Implementation from: https://github.com/MaximumEntropy/Seq2Seq-PyTorch
    Uses the following config: config_en_autoencoder_1_billion.json
    """
    def __init__(self, 
                 truncation_len=30,
                 adversarial_weight=.5,
                 baseline_weight = 0.99,
                 adversarial_checkpoint=None,
                 autoencoder_checkpoint=None, 
                 discriminator_checkpoint=None,
                 dataset=None):
        
        self.truncation_len = truncation_len
        self.adversarial_weight = adversarial_weight
        self.autoencoder_weight = 1 - adversarial_weight
        self.baseline_weight = baseline_weight
        self.dataset = dataset 
        self.vocab_size = len(self.dataset.vocab_encoder.word2index)
        self.pad_token_ind = self.dataset.vocab_encoder.word2index['<PAD>']
        self.batch_size = 10
        
        # Initialize the models.
        self.adversarial_model = Seq2SeqAutoencoder(
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
        
        self.autoencoder_model = Seq2SeqAutoencoder(
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
        
        # Restore the autoencoder and discriminator from the checkpoint if provided.
        self.autoencoder_model.load_state_dict(torch.load(open(autoencoder_checkpoint)))
        self.autoencoder_model.eval()
        self.discriminator_model = MultinomialNBDiscriminator(checkpoint=discriminator_checkpoint,
                                                        truncation_len=truncation_len)
        
        # Restore adversarial model from checkpoint if provided.
        if adversarial_checkpoint:
            self.restore_model(adversarial_checkpoint)
        
        # Initialize the distance module.
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # Initialize the optimizer.
        self.lr = 0.0002
        self.clip_c = 1
        self.optimizer = optim.Adam(self.adversarial_model.parameters(), lr=self.lr)
        
        
        # Loss criterion.
        weight_mask = torch.ones(self.vocab_size).cuda()
        weight_mask[self.pad_token_ind] = 0
        self.reward_baseline = None
        self.loss_criterion = self.adversarial_loss_criterion
        
        # Save the initial adversarial model.
        self.save_model()
    
    def adversarial_loss_criterion(self, decoder_logit, reward):
        # Baseline: use a simple EMA baseline.
        if self.reward_baseline is None:
            self.reward_baseline = reward.mean()
        else:
            self.reward_baseline = (self.baseline_weight * self.reward_baseline) + ((1 - self.baseline_weight) * reward.mean())

        # Advantage = reward - baseline.
        advantage = (reward - self.reward_baseline).detach() # Detach to avoid accumulating gradients.

        # Sum the log_probs of the sampled sentences over the timesteps:
        max_probs, _unused = decoder_logit.max(dim=-1)
        sum_max_probs = max_probs.sum(dim=-1)
        
        reduced_loss = advantage * sum_max_probs
        return reduced_loss.mean()
        
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
        input_lines = Variable(torch.LongTensor(np.fliplr(minibatch).copy()), 
                               requires_grad=False).cuda() # Reverse the input lines.
        return input_lines
    
    def perplexity(self):
        """Calculate the BLEU score."""
        pass
    
    def l2_normalize(self, x):
        norm = x.norm(p=2, dim=1, keepdim=True)
        x_normalized = x.div(norm.expand_as(x))
        return x_normalized
    
    def reward_function(self, input_lines_src, decoder_indices):
        """Calculate the reward."""
        # Get and penalize the autoencoder differences.
        src_encoding = self.autoencoder_model.encode(input_lines_src.detach())
        trg_encoding = self.autoencoder_model.encode(decoder_indices.detach())
        # Normalize the encodings and calculate the cos. similarity.
        autoencoder_difference = self.cos(src_encoding, trg_encoding)
        
        # Get the discriminator probs.
        discriminator_examples = decoder_indices.data.cpu().numpy()
        discriminator_probs = Variable(torch.FloatTensor(
            self.discriminator_model.get_adversarial_probs(discriminator_examples)), requires_grad=False).cuda()
        
        return (self.adversarial_weight * discriminator_probs) + (self.autoencoder_weight * autoencoder_difference)
        
    def train(self, dataset, epochs=2, write_checkpoint=1, monitor_loss=1, print_samples=1):
        examples, _ = dataset.examples(split="val", shuffled=True)
        num_examples, max_len = examples.shape
        
        for epoch in xrange(epochs):
            losses = []
            for iter_ind in xrange(0, num_examples, self.batch_size):
                self.adversarial_model.zero_grad()
                self.autoencoder_model.zero_grad()
                
                # Get a minibatch.
                input_lines_src = self.get_dataset_minibatch(examples, iter_ind, self.batch_size)
                
                # Get the adversarial decoder logits.
                decoder_logit = self.adversarial_model(input_lines_src)
                
                ### Calculate the reward from the inputs and outputs.
                decoder_indices = self.adversarial_model.decode_argmax(decoder_logit)
                reward = self.reward_function(input_lines_src, decoder_indices)      
                
                # Update the adversarial model.
                self.optimizer.zero_grad()
                loss = self.loss_criterion(decoder_logit, reward)
                losses.append(loss.data[0])
                loss.backward()
                self.optimizer.step()
                
                if iter_ind % monitor_loss == 0:
                    logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (epoch, iter_ind, np.mean(losses)))
                    losses = []
                    
                    # Memory check
                    #print "NUM OBJECTS: " + str(len([obj for obj in gc.get_objects() if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))]))
                
                if iter_ind % print_samples == 0:
                    # Print samples.
                    word_probs = decoder_indices.data.cpu().numpy()
                    output_lines_trg = input_lines_src.data.cpu().numpy()
                    real_probs = self.discriminator_model.get_adversarial_probs(output_lines_trg[:10])
                    pred_probs = self.discriminator_model.get_adversarial_probs(word_probs[:10])
                    for sentence_pred, sentence_real, real_prob, pred_prob in zip(word_probs[:10], 
                                                                                  output_lines_trg[:10],
                                                                                  real_probs,
                                                                                  pred_probs):
                        if pred_prob > real_prob:
                            decoded_real = dataset.vocab_encoder.decode_numpy(sentence_real[::-1])
                            decoded_pred = dataset.vocab_encoder.decode_numpy(sentence_pred)

                            logging.info('===============================================')
                            logging.info("REAL: " + str(real_prob) + " " + decoded_real)
                            logging.info("PREDICTED: " + str(pred_prob) + " " + decoded_pred)
                            logging.info('===============================================')
                            del decoded_real, decoded_pred
                    # Evaluate the samples and print ones where the probability increased.
                    
                    logging.info("Mean real probs: " + str(np.mean(real_probs)))
                    logging.info("Mean pred probs: " + str(np.mean(pred_probs)))
                    del real_probs, pred_probs, word_probs, output_lines_trg
                
                del input_lines_src, reward, decoder_logit, decoder_indices
                gc.collect()
            if epoch % write_checkpoint == 0:
                self.save_model()
        
    def evaluate(self, dataset, split, verbose=True):
        raise Exception("Not implemented")
    
    def save_model(self, 
                   checkpoint_dir='/cvgl2/u/catwong/cs332_final_project/checkpoints',
                   checkpoint_name='dancin_seq2seq_adversarial'):
        # Outputs a path that can be passed into the restore.
        checkpoint_file = checkpoint_name + '.model'
        full_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        torch.save(
            self.adversarial_model.state_dict(),
            open(full_checkpoint_path, 'wb')
        )
        return full_checkpoint_path
    
    def restore_model(self, checkpoint):
        self.adversarial_model.load_state_dict(torch.load(open(checkpoint)))

# Demo
if __name__ == "__main__":
    adversarial_weights = [.5, .8, .9, .95]
    truncation_len=30
    adversarial_weight = adversarial_weights[-1] # The weight to give to "fooling" the discriminator
    autoencoder_weight = 1 - adversarial_weight
    easy_dataset = True # Whether to use lower confidence spam.
    
    experiment_name = "%d_trunc_%d_adv_%d_auto_%d_easy_dancin" % (truncation_len, 
                                                                  adversarial_weight*100, 
                                                                  autoencoder_weight*100, 
                                                                  easy_dataset)
    print "Experiment: " + experiment_name    
    
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
    
    # If True, use the easy spam dataset composed of lower confidence scores.
    if easy_dataset:
        spam_dataset = SpamDataset(truncation_len=truncation_len, encoded_files=['encoded_spam_low_conf.txt'])
    else:
        spam_dataset = SpamDataset(truncation_len=truncation_len, encoded_files=['encoded_spam.txt'])
    
    # Get all of the checkpoints.
    checkpoints_dir = '/cvgl2/u/catwong/cs332_final_project/checkpoints'
    discriminator_checkpoint_files = ["_multinomial_nb_model.pkl", "_multinomial_nb_tf_transformer.pkl"]
    discriminator_checkpoint = [os.path.join(checkpoints_dir, str(truncation_len) + filename) 
                                for filename in discriminator_checkpoint_files]
    autoencoder_checkpoint_file = str(truncation_len) + "_seq2seq_autoencoder.model"
    autoencoder_checkpoint = os.path.join(checkpoints_dir, autoencoder_checkpoint_file)
                        
    adversarial_checkpoint_file = experiment_name + ".model"
    adversarial_checkpoint =  os.path.join(checkpoints_dir, autoencoder_checkpoint_file)   
                        
    adversarial = DancinSeq2SeqAdversarial(truncation_len=30, 
                                           adversarial_weight=adversarial_weight,
                                           adversarial_checkpoint=adversarial_checkpoint,
                                           autoencoder_checkpoint=autoencoder_checkpoint, 
                                           discriminator_checkpoint=discriminator_checkpoint,
                                           dataset=spam_dataset)
    adversarial.train(
        dataset=spam_dataset, 
        epochs=1000, 
        write_checkpoint=100,
        monitor_loss=100,
        print_samples=100)
    #checkpoint = adversarial.save_model(checkpoint_name=experiment_name)
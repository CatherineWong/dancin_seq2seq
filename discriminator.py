"""
discriminator.py - Discriminator classes.

Classes:
    Discriminator: a general discriminator interface.
    MultinomialNB: a multinomial NaiveBayes subclass.
"""
from __future__ import division

import os
import numpy as np
import scipy
import scipy.stats
import sklearn
import sklearn.feature_extraction, sklearn.naive_bayes, sklearn.metrics, sklearn.externals
from collections import defaultdict, Counter

from dataset import SpamDataset

# The number of terms, including special tokens, in the final vocabulary.
TRAINING_VOCAB_SIZE = {
    100: 4480,
    30: 4628
} 


class Discriminator(object):
    """
    Discriminator: a general discriminator class.
    """
    def __init__(self, checkpoint=None):
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

class MultinomialNBDiscriminator(Discriminator):
    """
    MultinomialNB: Multinomial Naive Bayes Classifier w. alpha=1.0
    
    Trained using TF-IDF features.
    """
    def __init__(self, truncation_len=100, checkpoint=None):
        Discriminator.__init__(self, checkpoint)
        self.truncation_len=truncation_len
        if not checkpoint:
            self.model = sklearn.naive_bayes.MultinomialNB()
        else:
            self.restore_model(checkpoint)
    
    def examples_to_term_doc(self, examples):
        """
        Converts a numerically-encoded examples matrix into a sparse term-documents matrix.
        """
        num_terms = TRAINING_VOCAB_SIZE[self.truncation_len]
        all_row_inds = all_col_inds = all_data = None
        for row_ind, example in enumerate(examples):
            if row_ind % 5000 == 0:
                print "Generating term-docs matrix: %d of %d" %(row_ind, len(examples))
            itemfreqs = scipy.stats.itemfreq(example).T
            # Column indices: the term indices in that document.
            col_inds = itemfreqs[0]
            # Data: the counts of the terms in that document.
            data = itemfreqs[1]
            # Row indices: the current document, for each of the terms in that document.
            row_inds = np.ones(itemfreqs.shape[1], dtype=np.int) * row_ind

            # Concatenate to the existing data.
            if all_row_inds is None:
                all_row_inds = row_inds
                all_col_inds = col_inds
                all_data = data
            else:
                all_row_inds = np.append(all_row_inds, row_inds)
                all_col_inds = np.append(all_col_inds, col_inds)
                all_data = np.append(all_data, data)

        num_docs = len(examples)
        return scipy.sparse.csr_matrix((all_data, (all_row_inds, all_col_inds)), shape=(num_docs, num_terms))

    def train(self, dataset):
        examples, labels = dataset.examples(split='train', shuffled=True)
        
        # Silly way to compute sparse doc term matrix from examples matrix by converting it back into "strings".
        self.train_counts = self.examples_to_term_doc(examples)
        
        # Featurize using TFIDF.
        self.tf_transformer = sklearn.feature_extraction.text.TfidfTransformer()
        X_transformed = self.tf_transformer.fit_transform(self.train_counts)
        
        # Fit the model to TFIDF counts.
        self.model.fit(X_transformed, labels)
    
    def calculate_roc_auc(self, probs, labels):
        # Probability estimates of the positive class.
        pos_probs = probs[:, 1]
        return sklearn.metrics.roc_auc_score(labels, pos_probs)
    
    def evaluate(self, dataset, split, verbose=True):
        # Get the test or validation examples.
        examples, labels = dataset.examples(split=split, shuffled=True)
        doc_terms = self.examples_to_term_doc(examples)
        X_transformed = self.tf_transformer.transform(doc_terms)
        
        # Evaluate the model.
        probs = self.model.predict_proba(doc_terms)
        predicted = np.argmax(probs, axis=1)
        
        # Mean accuracy.
        mean_accuracy = np.mean(predicted == labels)
        print "Mean_accuracy: %f" % mean_accuracy
        
        # ROC-AUC Score.
        roc_auc = self.calculate_roc_auc(probs, labels)
        print "ROC AUC: %f" % roc_auc
    
    def save_model(self, 
                   checkpoint_dir='/cvgl2/u/catwong/cs332_final_project/checkpoints',
                   checkpoint_name='multinomial_nb'):
        # Separately pickles the model and the transformer.
        checkpoint = os.path.join(checkpoint_dir, str(self.truncation_len) + "_" + checkpoint_name)
        sklearn.externals.joblib.dump(self.model, checkpoint + "_model.pkl")
        sklearn.externals.joblib.dump(self.tf_transformer, checkpoint + "_tf_transformer.pkl")
        return [checkpoint + "_model.pkl", checkpoint + "_tf_transformer.pkl"]
    
    def restore_model(self, model_checkpoints):
        self.model = sklearn.externals.joblib.load(model_checkpoints[0])
        self.tf_transformer = sklearn.externals.joblib.load(model_checkpoints[1])

# Demonstration models.
if __name__ == "__main__":
    for truncation_len in [30, 100]:
        print "Now on truncation_len: " + str(truncation_len)
        spam_dataset = SpamDataset(truncation_len=truncation_len)

        checkpoints_dir = '/cvgl2/u/catwong/cs332_final_project/checkpoints'
        checkpoint_files = ["_multinomial_nb_model.pkl", "_multinomial_nb_tf_transformer.pkl"]
        checkpoint = [os.path.join(checkpoints_dir, str(truncation_len) + filename) for filename in checkpoint_files]
        new_discriminator = MultinomialNBDiscriminator(checkpoint=checkpoint, truncation_len=truncation_len)
        new_discriminator.evaluate(spam_dataset, 'val')
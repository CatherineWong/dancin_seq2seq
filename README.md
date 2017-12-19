# DANCin Seq2Seq: Dueling Adversarial Neural Classification in Seq2Seq Models
CS332 (Advanced Topics in RL) Final Project, Autumn 2017

Fool me once, shame on me. Fool me twice, maybe you're using an adversarial discriminator<--autoencoder-->generator formulation to
generate spam that looks a lot less like spam!

Technical details and experimental results in this paper:
  1. DANCin SEQ2SEQ: Fooling Text Classifiers with Adversarial Text Example Generation: https://arxiv.org/pdf/1712.05419.pdf

Huge thanks to:
1. Adversarial Learning for Neural Dialogue Generation: https://arxiv.org/pdf/1701.06547.pdf
This policy gradient GAN-like formulation for text example generation borrows heavily from this paper.

2. The Seq2Seq Autoencoder implemented by MaximumEntropy: https://github.com/MaximumEntropy/Seq2Seq-PyTorch

# MSCGPCN
Multi-view Silhouette Constraint Generative Point Completion Network. The network based GAN, including u-net structure generator, point cloud discriminator and multi-view silhouette discriminator.

Train_bank.py:

  A script to train a neural network, including reading the data, preprocessing the data, training, and calculating the loss.

data_utils.py:

  A computational method of processing data.

model_Net.py:

  Various network structures were defined, including the generator of the designed u-net structure and the multi-view projection silhouuette image discriminator.

shapenet_part_loarder.py:

  Loading dataset.

show_bank.py:

  Visualization of the validation set.

utils.py:

  The functions of data processing and training.

# MSCGPCN
Multi-view Silhouette Constraint Generative Point Completion Network. The network based GAN, including u-net structure generator, point cloud discriminator and multi-view silhouette discriminator.

visual_completion_results.mp4:
  This is a video showing the results corresponding to Section 3.2 of the paper, "The completion results of the plant point cloud validation set".
  It is designed to facilitate the observation of the completion results from a three-dimensional perspective.
  In the video, the brown points represent the incomplete point cloud input to the network, 
      the blue points represent the point cloud of the actual missing area, 
      and the red points represent the point cloud of the missing area predicted to be completed by the network.

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

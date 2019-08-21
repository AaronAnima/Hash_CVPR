import numpy as np
import tensorlayer as tl

class FLAGS(object):
    def __init__(self):
        ''' For training'''
        self.n_epoch = 10 # "Epoch to train [25]"
        self.z_dim = 100 # "Dim of noise value]"
        self.h_dim = 16 # Dim of hash code
        self.c_dim = 3 # "Number of image channels. [3]")
        self.learning_rate = 0.0002 # "Learning rate of for adam [0.0002]")
        self.beta1 = 0.5 # "Momentum term of adam [0.5]")
        self.batch_size_train = 64 # "The number of batch images [64]")
        self.save_step = 500 # "The interval of saveing checkpoints. [500]")
        self.dataset = "CIFAR_10" # "The name of dataset [CIFAR_10, MNIST]")
        self.checkpoint_dir = "checkpoint" # "Directory name to save the checkpoints [checkpoint]")
        self.sample_dir = "samples" # "Directory name to save the image samples [samples]")
        self.img_size_h = 32 # Img height
        self.img_size_w = 32  # Img width
        self.lambda_minEntrpBit = [0.001, 0.01] # Lambda for minEntropy loss
        self.lambda_HashBit = 1 # Lambda for HashBit loss
        self.lambda_L2 = 0.1 # lambda for L2 loss
        self.init_learning_rate = 0.0003 # initial learning rate
        self.max_learning_rate = 0.0009 # Max(final) learning rate
        self.eval_step = 781 # Evaluation freq during training
        self.hash_loss_threshold = 0.1

        ''' For eval '''
        self.eval_epoch_num = 5
        self.eval_print_freq = 5000 #
        self.retrieval_print_freq = 200
        self.eval_sample = 1000 # Query num for mAP matrix
        self.nearest_num = 1000 # nearest obj num for each query
        self.batch_size_eval = 1  # batch size for every eval





flags = FLAGS()

tl.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tl.files.exists_or_mkdir(flags.sample_dir) # save generated image
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
import math

logging.basicConfig(level=logging.INFO)

# List of TODOs for later
# 1. Experiment with other optimizers like AdamOptimizer


# TODO: Experiment with other optimizers like AdamOptimizer
def get_optimizer:
    return tf.train.GradientDescentOptimizer


# TODO: Define the encoder class(simplify it because maybe V1 don't use LSTMs yet


# TODO: Define the decoder class(simplify it because maybe V1 don't use LSTMs yet


# TODO: Define the TensorFlowQuestionAnswerSystem
class TensorFlowQuestionAnswerSystem(object):
    """
    Initialize the TensorFlow model
    
    Input
    encoder: constructed in train.py
    decoder: constructed in train.py
    args: extra arguments defining the details of the model/system
    """
    # TODO : pass in the self.embed_path correctly (not sure what that is right now)
    
    def __init__(self, encoder, decoder, fn_convert_index_to_word, args):
        # ------ Define encoder, decoder, and fn_convert_index_to_word ------
        self.encoder = encoder
        self.decoder = decoder
        self.fn_convert_index_to_word = fn_convert_index_to_word

        # ------ Define other fields of the model/system ------
        self.max_passage_length = args.max_passage_length
        self.max_question_length = args.max_question_length
        #self.embedding_size = args.embedding_size
        self.embed_path = args.embed_path
        self.learning_rate = args.learning_rate
        self.num_epochs = args.epochs
        #self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        #self.max_gradient_norm = args.max_gradient_norm
        #self.train_dir = args.train_dir
        #self.saved_name = args.saved_name
        #self.eval_num_samples = args.eval_num_samples
        #self.val_and_save_num_batches = args.val_and_save_num_batches
        #self.val_cost_frac = args.val_cost_frac
        self.size_train_dataset = args.size_train_dataset
        #self.sigma_threshold = args.sigma_threshold

        # ------ Load the embeddings (word vectors) with GLoVE -------
        self.pretrained_embeddings = np.load(self.embed_path)['glove']


        # ------ Set up teh placeholder variables ------
        # Matrix with dimensions (batch_size by maximum passage length)
        self.passages_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        # Matrix with dimensions (batch_size by maximum question length)
        self.questions_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        # Matrix with dimensions (batch_size by 2) where the second dimension is a binary indicator for each word. 0 represents the score for when word is not part of the answer and 1 represents the score when it is
        # TODO: confirm if this is score in fact or something else
            self.answers_placeholder = tf.placeholder(tf.int32, shape=(None, None))

        # Placeholders for bidirectional lstm
        #self.passage_sequence_lengths = tf.placeholder(tf.int32, [None])
        #self.question_sequence_lengths = tf.placeholder(tf.int32, [None])
        # Create global step counter so we can track and save how many batches
        # we've completed.
        #self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # The ordering of indices in the currently running shuffled batch
        #self.idxs = tf.Variable(tf.zeros(self.size_train_dataset, dtype=tf.int32), \
                                name='idxs', trainable=False)

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
        self.preds = self.setup_predictions() # Creates embeddings and prediction
        self.loss = self.setup_loss(self.preds) # Creates loss computation
        self.train_op, self.grad_norm = self.setup_learning(self.loss) # Creates optimizer i.e. updates parameters in model
        
        # Create model saver
        self.saver = tf.train.Saver()







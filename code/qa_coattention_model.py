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

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim # This represents embedding size

    def encode(self, inputs, sequence_lengths, keep_prob, masks=None, encoder_state_input=None):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        : param passage_sequence_lengths: This is the sequence length for each passage in the batch.
            They're all the same and correspond to max_length_passage
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        # Inputs is tuple
        # Inputs = (passages_batch, questions_batch)
        passages, questions = inputs
        # Sequence lengths is tuple
        # Sequence_lengths = (passage_lengths, question_lengths)
        passage_sequence_lengths, question_sequence_lengths = sequence_lengths

        # We assume passages_batch is (None, max_length_passage, embedding dim) and represents the word embedding of the passages
        # We assume questions_batch is (None, max_length_question, embedding dim) and represents the word embedding of the questions
        # Each index in the second dimension represents the word at that index
        # TODO: add mask if we want to use the final state. step-by step is probably chill
        # See: https://piazza.com/class/iw9g8b9yxp46s8?cid=2153

        # Our model is the following:
        #   TODO: scale passage word embeddings with relevancy scores based on the maximum dot product between a given passage word embedding and each question word embedding
        #   run bid-rectional LSTM over the passage. Concatenate forward and backward vectors at each word/time-step
        #   run bid-rectional LSTM over the question. Concatenate forward and backward vectors for the last word
        #   for each time-step in passage, concatenate state vector with the vector above

        # Scale passage word embeddings with relevancy scores
        with vs.variable_scope("Relevancy-Scaling"):
            # take into account padding issues
            # P = [p1 p2 .. pM]; Q = [q1 q2 .. qN]
            # normalize P and Q by columns, get P' and Q'
            # C = (P')^T * Q' is cosine similarity
            # r_i of p_i = row_max(C)
            # create diagonal matrix of r_is R, get P~ = R * P
            normalized_passages = tf.nn.l2_normalize(x=passages, dim=2) # [batch_size, max_passage_length, embedding_size]
            normalized_questions = tf.nn.l2_normalize(x=questions, dim=2) # [batch_size, max_question_length, embedding_size]
            transposed_normalized_questions = tf.transpose(a=normalized_questions, perm=(0, 2, 1)) # [batch_size, embedding_size, max_question_length]
            cosine_similarity = tf.matmul(normalized_passages, transposed_normalized_questions) # [batch_size, max_passage_length, max_question_length]
            relevancy_scores = tf.reduce_max(input_tensor=cosine_similarity, axis=2) # [batch_size, max_passage_length]
            relevancy_scores = tf.expand_dims(input=relevancy_scores, axis=2) # [batch_size, max_passage_length, 1]
            relevancy_scores = tf.tile(input=relevancy_scores, multiples=[1, 1, self.vocab_dim]) # [batch_size, max_passage_length, embedding_size]
            passages = tf.multiply(relevancy_scores, passages) # [batch_size, max_passage_length, embedding_size]

        # Generate bi-lstm for passage
        with vs.variable_scope("Passage-Bi-LSTM"):
            # First pass, we just want to run a bi lateral LSTM over each passage in the batch
            # Create forward direction cell
            with vs.variable_scope('forward'):
                p_lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.size, forget_bias=1.0, state_is_tuple=True)
            # Create backward cell
            with vs.variable_scope('backward'):
                p_lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.size, forget_bias=1.0, state_is_tuple=True)

            # Create bilateral LSTM

            p_outputs, p_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=p_lstm_fw_cell, cell_bw = p_lstm_bw_cell, \
                inputs = passages,  dtype=tf.float64, scope="Passage-Bi-LSTM", sequence_length=passage_sequence_lengths)
            # Concatenate the output_fw and output_bw at each time-step for each input in batch
            # Outputs[0] corresponds to the forward output state at each time step
            # Outputs[1] corresponds to the backward otuput state at each time step
            p_concat_outputs = tf.concat(2, [p_outputs[0], p_outputs[1]]) # [batch_size, max_passage_length, 2 * hidden_size]
            p_concat_outputs_w_dropout = tf.nn.dropout(p_concat_outputs, keep_prob)

        # Generate bi-lstm for question
        with vs.variable_scope("Question-Bi-LSTM"):
            # First pass, we just want to run a bi lateral LSTM over each question in the batch
            # Create forward direction cell
            with vs.variable_scope('forward'):
                q_lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.size, forget_bias=1.0, state_is_tuple=True)
            # Create backward cell
            with vs.variable_scope('backward'):
                q_lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.size, forget_bias=1.0, state_is_tuple=True)
            # Create bilateral LSTM
            q_outputs, q_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=q_lstm_fw_cell, cell_bw = q_lstm_bw_cell, \
                inputs = questions,  dtype=tf.float64, scope="question-Bi-LSTM", sequence_length=question_sequence_lengths)

            # Only concat the forward state for the last time step and backward state for first time step
            # Outputs[0] corresponds to the forward output state at each time step
            # Outputs[1] corresponds to the backward otuput state at each time step

            # TODO: modify below to actually get all the concatenations of the hidden states; see above
            q_concat_outputs = tf.concat(2, [q_outputs[0], q_outputs[1]]) # [batch_size, max_question_length, 2 * hidden_size]
            q_concat_outputs_w_dropout = tf.nn.dropout(q_concat_outputs, keep_prob)

        with vs.variable_scope("Co-Attention-Summaries"):
            # take into account padding issues
            # P = [P~ | null]; Q = [q1 q2 .. qN | null]
            # C = (P~^T)Q
            # A^Q = softmax(C)
            # A^P = softmax(C^T)
            q_concat_outputs = tf.transpose(a=q_concat_outputs_w_dropout, perm=(0, 2, 1)) # [batch_size, 2 * hidden_size, max_question_length]
            transposed_p_concat_outputs = p_concat_outputs_w_dropout
            p_concat_outputs = tf.transpose(a=p_concat_outputs_w_dropout, perm=(0, 2, 1)) # [batch_size, 2 * hidden_size, max_passage_length]
            raw_coattention = tf.matmul(transposed_p_concat_outputs, q_concat_outputs) # [batch_size, max_passage_length, max_question_length]
            coattention = tf.nn.softmax(logits=raw_coattention, dim=-1) # [batch_size, max_passage_length, max_question_length]; normalized across last dimension
            alt_coattention = tf.transpose(a=tf.nn.softmax(logits=raw_coattention, dim=1), perm=(0, 2, 1)) # [batch_size, max_question_length, max_passage_length]
            document_summaries = tf.matmul(p_concat_outputs, coattention) # [batch_size, 2 * hidden_size, max_question_length]
            question_document_summaries = tf.concat(1, [q_concat_outputs, document_summaries]) # [batch_size, 4 * hidden_size, max_question_length]
            summaries = tf.matmul(question_document_summaries, alt_coattention) # [batch_size, 4 * hidden_size, max_passage_length]

            # create inputs for next scope, the Bi-LSTM
            passages_summaries = tf.concat(1, [p_concat_outputs, summaries]) # [batch_size, 6 * hidden_size, max_passage_length]
            passages_summaries = tf.transpose(a=passages_summaries, perm=(0, 2, 1)) # [batch_size, max_passage_length, 6 * hidden_size]

        with vs.variable_scope("Co-Attention-Bi-LSTM"):
            # Create forward direction cell
            with vs.variable_scope('forward'):
                u_lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.size, forget_bias=1.0, state_is_tuple=True)
            # Create backward cell
            with vs.variable_scope('backward'):
                u_lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.size, forget_bias=1.0, state_is_tuple=True)
            # Create bilateral LSTM
            u_outputs, u_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=u_lstm_fw_cell, cell_bw = u_lstm_bw_cell, \
                inputs = passages_summaries, dtype=tf.float64, scope="co-attention-Bi-LSTM", sequence_length=passage_sequence_lengths)

        u_concat_outputs = tf.concat(2, [u_outputs[0], u_outputs[1]]) # [batch_size, max_passage_length, 2 * hidden_size]
        u_concat_outputs_w_dropout = tf.nn.dropout(u_concat_outputs, keep_prob=keep_prob)

        return u_concat_outputs_w_dropout # [batch_size, max_passage_length, 2 * hidden_size]


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    # Creates a fully connected layer
    # Assumes inputs is (batch_size, passage_length, knowledge_rep_size)
    # We return (batch_size, passage_length, 1)
    def create_fc_layer(self, inputs, max_len_passage, name_space):
        batch_size, max_length, knowledge_size = inputs.get_shape().as_list()

        # Create variable under the specified namespace
        with tf.variable_scope(name_space):
            # Create weight matrix
            U = tf.get_variable("U",
                                [knowledge_size, 1],
                                dtype=tf.float64,
                                initializer=tf.contrib.layers.xavier_initializer())

            # Create bias vector
            b = tf.get_variable("b",
                                [1],
                                dtype=tf.float64,
                                initializer=tf.constant_initializer(0.0))

        # Since max_len_passage is dynamically computed, we cannot iterate over
        # every time step in the tensor  in order to compute the prediction.
        # Instead, we reshape the tensor into a 2D matrix so that we can compute
        # predictions for all time steps with one matrix multiplication.
        # NOTE: Assuming tf.reshape unrolls dimensions in the exact opposite order
        # it rolls dimensions (initial tests on numpy seem to indicate this)
        d_inputs_reshaped = tf.reshape(inputs, [-1, knowledge_size])

        outputs = tf.matmul(d_inputs_reshaped, U) + b

        # Our outputs are of size (batch_size*max_len_passage, output_size), we needed
        # to reshape them back so they are grouped by timestep
        outputs = tf.reshape(outputs, [-1, max_len_passage])
        return outputs


    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: a Tensor of size (batch_size, max_length_passage, knowledge_size)
        :return:
        """
        # We take in (batch_size, max_passage_length, XXX)
        # We compute two separate fully connected layers
        # We return tuple: ( (batch_size, max_passage_length, 1), (batch_size, max_passage_length, 1) )
        # We assume the first item is distribution over start indexes. The second is for end indexes

        # Get what the length of each passage is since it's dynamic
        max_len_passage = tf.shape(knowledge_rep)[1]

        # Create layer for start indices
        a_t_outputs = self.create_fc_layer(knowledge_rep, max_len_passage, "start_index_prediction")
        # Create layer for end indices
        e_t_outputs = self.create_fc_layer(knowledge_rep, max_len_passage, "end_index_prediction")
        # Return Tuple
        return (a_t_outputs, e_t_outputs)

class QASystem(object):
    # TODO: automatic padding?? automatic batching??
    # TODO: find more efficient way to do loss with the answers so we don't have to expand them out
    # TODO: make pre-processing more efficient
    # TODO: decide on max lengths
    # TODO: how to deal with things > max length
    def __init__(self, encoder, decoder, rev_vocab, args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== Setup hyper parameters =======
        self.max_length_passage = args.max_passage_length # The length of each passage with padding
        self.max_length_question = args.max_question_length # The length of each question with padding
        self.embedding_size = args.embedding_size # The size of the Glove Vectors
        self.embed_path = args.embed_path # Where the Glove vectors live
        self.start_learning_rate = args.start_learning_rate # The learning rate we want to start with
        self.learning_decay_rate = args.learning_decay_rate # Rate at which learning rate decays exponentially.
        self.num_decay_steps = args.num_decay_steps
        self.epochs = args.epochs # Number of epochs to use. Epoch is one pass over all data
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size # Batch size
        self.max_gradient_norm = args.max_gradient_norm # The max gradient norm we use for clipping
        self.train_dir = args.train_dir # Where to save weights to
        self.saved_name = args.saved_name # What the base name to use to save weights
        self.eval_num_samples = args.eval_num_samples # Number of samples to use with evaluate_answer
        self.val_num_batches = args.val_num_batches
        self.val_cost_frac = args.val_cost_frac # What fraction of validation to use when computing this
        self.size_train_dataset = args.size_train_dataset
        self.l2_lambda = args.l2_lambda
        self.dropout = args.dropout
        self.num_keep_checkpoints = args.num_keep_checkpoints # Number of saved models we keep (defaults to all saved models)

        # ==== Set encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.rev_vocab = rev_vocab

        # ==== Load any data we need ========
        # First load word embeddings
        self.pretrained_embeddings = np.load(self.embed_path)['glove'] # We assume it's glove

        # ==== set up placeholder tokens ========

        # The first dimension is the batch_size and second dimension represents maximum passage length
        self.passages_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        # The first dimension is the batch_size and second dimension represents maximum question length
        self.questions_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        # The first dimension is the batch_size and second dimension represents binary indicator for each word
        # 0 represents the word is not the start index. 1 represents it is.
        self.a_t_placeholder = tf.placeholder(tf.int32, [None])
        self.e_t_placeholder = tf.placeholder(tf.int32, [None]) # Same format as above but for end index

        # Need masks for both passages and questions
        self.mask_passage_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.mask_question_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        # Placeholders for bidirectional lstm
        # TODO: Question: is there a better way of doing this??
        # This is constant list of batch_size where each index represents the number of words in the passage
        self.passage_sequence_lengths = tf.placeholder(tf.int32, [None])
        self.question_sequence_lengths = tf.placeholder(tf.int32, [None])

        # Keep probability for dropout after non-linearities
        self.keep_prob = tf.placeholder(tf.float64, shape=())

        # Create global step counter so we can track and save how many batches
        # we've completed.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # The ordering of indices in the currently running shuffled batch
        self.idxs = tf.Variable(tf.zeros(self.size_train_dataset, dtype=tf.int32), \
                                         name='idxs', trainable=False)

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.preds = self.setup_predictions() # Creates embeddings and prediction
            self.loss = self.setup_loss(self.preds) # Creates loss computation
            self.train_op, self.grad_norm, self.new_grad_norm = self.setup_learning(self.loss) # Creates optimizer i.e. updates parameters in model

        # Create model saver
        # Do this after all operations have been created
        # SUPER IMPORTANT!!!!!
        # Since we start at 0 when we initialize, if we run train again without changing the base file name, we will overload previously saved info
        self.saver = tf.train.Saver(max_to_keep=self.num_keep_checkpoints)
        # ==== set up training/updating procedure ====


    def setup_predictions(self):
        # This generates predictions operations that we feed into loss
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        # Get embeddings
        passage_embeddings, question_embeddings = self.setup_embeddings()

        # Setup encoder inputs
        encoder_inputs = (passage_embeddings, question_embeddings)
        sequence_lengths = (self.passage_sequence_lengths, self.question_sequence_lengths)
        # Assemble encoder operations on computation graph
        # We take our initial embeddings/input and generate some sort of representation
        encode_output = self.encoder.encode(encoder_inputs, sequence_lengths, self.keep_prob)

        # Assemble decoder operations on computation graph
        # We take whatever output from encoder and generate probabilities/predictions from it
        decode_output = self.decoder.decode(encode_output)

        return decode_output

    def setup_loss(self, preds):
        # This sets up the loss graph operation
        """
        Set up your loss computation here
        :return:
        """
        # TODO: Question: would it be faster to simply sum over the log of the indices where we have the correct answer?
        with vs.variable_scope("loss"):
            # We assume loss is a tuple of (a_t predictions, e_t predictions)
            # We also only calculate the loss at the end-points
            # We assume we have two answers placeholders. One that list indices for start indices. One for end indices
            # Because we want to do softmax across time_steps, we permute to (batch_size, 2, time_steps) for both
            # Add in loss
            cross_a_t = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds[0], labels=self.a_t_placeholder)
            cross_e_t = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds[1], labels=self.e_t_placeholder)

            weights = [var for var in tf.trainable_variables() if not ("/B:" in var.name or "/b:" in var.name)]
            l2 = self.l2_lambda * tf.add_n([tf.nn.l2_loss(weight) for weight in weights])

            loss = tf.reduce_mean(cross_a_t) + tf.reduce_mean(cross_e_t) + l2
            return loss

    def setup_learning(self, loss):
        # Think function links the loss to updating the parameters
        # We add gradient clipping because LSTMS can blow up
        optimizer = tf.train.AdamOptimizer(learning_rate=self.start_learning_rate)

        gradients, v = zip(*optimizer.compute_gradients(loss))
        grad_norm = tf.global_norm(gradients)
        gradients, new_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm, use_norm=grad_norm)
        train_op = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)
        #train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return train_op, grad_norm, new_norm

    def setup_embeddings(self):
        # This turns input placeholder into word embeddings
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        # TODO: Question: do we have to use get variable?
        # TODO: Question: how does this deal with words not being in Glove??
        # TODO: check if this is even remotely right
        with vs.variable_scope("embeddings"):
            embs = tf.Variable(self.pretrained_embeddings)
            passage_t = tf.nn.embedding_lookup(embs, self.passages_placeholder)

            # Dynamically pull passage length because self.max_length_passage is only used
            # during training.
            max_len_passage = tf.shape(self.passages_placeholder)[1]
            # Get passage embeddings
            passage_embeddings = tf.reshape(passage_t, [-1, max_len_passage, self.embedding_size])
            question_t = tf.nn.embedding_lookup(embs, self.questions_placeholder)

            # Dynamically pull question length because self.max_length_question is only used
            # during training.
            max_len_question = tf.shape(self.questions_placeholder)[1]
            # Get question embeddings
            question_embeddings = tf.reshape(question_t, [-1, max_len_question, self.embedding_size])
            return passage_embeddings,question_embeddings

    def optimize(self, session, train_x, train_y):
        # This is just boilerplate code for passing in feed dictionary and then calling
        # the graph operations for setup_leanring and also setup_system (to get the actual predictions)

        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        # Unpack input. We assume all are passed in
        passages_batch, questions_batch, passage_masks_batch, question_masks_batch = train_x
        # Populate feed dict
        input_feed = {
            self.passages_placeholder : passages_batch,
            self.questions_placeholder : questions_batch,
            self.mask_passage_placeholder : passage_masks_batch,
            #self.mask_question_placeholder : question_masks_batch,
            self.passage_sequence_lengths : np.sum(passage_masks_batch.astype(int), 1),
            self.question_sequence_lengths: np.sum(question_masks_batch.astype(int), 1),
            self.a_t_placeholder : train_y[0], # Assumes train_y is a tuple
            self.e_t_placeholder : train_y[1],
            self.keep_prob : self.dropout
        }
        # Specify operations and values we want back
        output_feed = [self.train_op, self.loss, self.preds, self.grad_norm, self.new_grad_norm]
        train, loss, outputs, grad_norm, new_grad_norm = session.run(output_feed, input_feed)

        return loss, outputs, grad_norm, new_grad_norm

    def test(self, session, dataset):
        # This calls the setup_loss graph operation to compute a cost for the inputted validation set

        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        passages_batch, questions_batch, answers_batch, passage_masks_batch, \
            question_masks_batch = dataset

        input_feed = {
            self.passages_placeholder : passages_batch,
            self.questions_placeholder : questions_batch,
            self.mask_passage_placeholder : passage_masks_batch,
            #self.mask_question_placeholder : question_masks_batch,
            self.passage_sequence_lengths : np.sum(passage_masks_batch.astype(int), 1),
            self.question_sequence_lengths: np.sum(question_masks_batch.astype(int), 1),
            self.a_t_placeholder : answers_batch[0], # Assumes answer is a tuple
            self.e_t_placeholder : answers_batch[1],
            self.keep_prob : 1.0
        }

        output_feed = [self.loss, self.preds]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        # This just runs the graph operation for setup_preds
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        # Unpack input. We assume all are passed in
        passages_batch, questions_batch, answers_batch, passage_masks_batch, question_masks_batch = test_x
        # Populate feed dict
        input_feed = {
            self.passages_placeholder : passages_batch,
            self.questions_placeholder : questions_batch,
            self.mask_passage_placeholder : passage_masks_batch,
            self.mask_question_placeholder : question_masks_batch,
            self.passage_sequence_lengths : np.sum(passage_masks_batch.astype(int), 1),
            self.question_sequence_lengths: np.sum(question_masks_batch.astype(int), 1),
            self.keep_prob : 1.0
        }
        # Specify operations and values we want back
        output_feed = [self.preds]
        outputs = session.run(output_feed, input_feed)[0]
        return outputs

    def answer(self, session, test_x, raw_predictions_batch=None):
        """
        Returns, for each test example, a list of indices, each representing
        a place where our model believes a word is part of the final answer.

        :return: Tensor of size [batch_size, self.max_length_passage]
        """
        # assuming output of ([batch_size, self.max_length_passage, 2], [batch_size, self.max_length_passage, 2]);
        # these are logits
        if raw_predictions_batch is None:
            logging.info("No predictions given so we have to calculate")
            raw_predictions_batch = self.decode(session, test_x)
        # convert logits to probabilities via softmax
        raw_predictions_a_t = raw_predictions_batch[0] # Get start index distribution
        raw_predictions_e_t = raw_predictions_batch[1] # Get end index distribution
        num_predictions = len(raw_predictions_a_t)
        # Now we softmax
        predictions_a_t = tf.nn.softmax(raw_predictions_a_t).eval() # Perform softmax over entire passage
        predictions_e_t = tf.nn.softmax(raw_predictions_e_t).eval() # Perform softmax over entire passage

        #predictions_batch = tf.nn.softmax(raw_predictions_batch).eval()
        cleaned_predictions_batch = []
        contexts_masks, _, _, _, _ = test_x
        # range of [0, 1, ..., self.max_length_passage - 1]
        for i in xrange(num_predictions):
            related_context_mask = contexts_masks[i]
            end_of_passage = int(np.sum(related_context_mask))
            prediction_a_t = predictions_a_t[i] # Get start index distribution for passage. Should be max_passage X 2 array
            prediction_e_t = predictions_e_t[i] # Same for end index
            # Note, we assume second index is the index corresponding to probability of being start index.
            start_index = np.argmax(prediction_a_t[0:end_of_passage]) # Find the word with the largest prob of being start index
            if start_index == end_of_passage - 1: # Start index is the last word
                end_index = start_index
            else:
                # Set end word as the word after the start index with largest probability of being end index
                end_index = np.argmax(prediction_e_t[start_index:end_of_passage]) + start_index
            # Generate range between start and end index
            answer_indices = range(start_index, end_index+1)
            cleaned_predictions_batch.append(answer_indices)

        return cleaned_predictions_batch

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        passages, questions, answers, passage_masks, question_masks = valid_dataset

        # We assume answers is a tuple
        a_t = answers[0]
        e_t = answers[1]

        losses = []
        predictions = None

        batch_size = self.batch_size

        for batch_idx in range(0, len(passages), batch_size):
            passages_batch = passages[batch_idx:batch_idx+batch_size]
            questions_batch = questions[batch_idx:batch_idx+batch_size]
            a_t_batch = a_t[batch_idx:batch_idx+batch_size]
            e_t_batch = e_t[batch_idx:batch_idx+batch_size]
            passage_masks_batch = passage_masks[batch_idx:batch_idx+batch_size]
            question_masks_batch = question_masks[batch_idx:batch_idx+batch_size]

            marginal_loss, marginal_predictions = self.test(sess, \
                (passages_batch, questions_batch, (a_t_batch, e_t_batch), \
                    passage_masks_batch, question_masks_batch))

            # (longA, longB), to append (newA, newB)
            # TODO: might need to change this
            losses.append(marginal_loss)
            if predictions is None:
                predictions = marginal_predictions
            else:
                a_t_batches = tf.concat(0, [predictions[0], marginal_predictions[0]])
                e_t_batches = tf.concat(0, [predictions[1], marginal_predictions[1]])
                predictions = (a_t_batches, e_t_batches)

        loss = tf.reduce_mean(losses)
        return loss, predictions

    def evaluate_answer(self, session, dataset, rev_vocab = None, raw_predictions_batch = None, sample=3, log=False):
        # Assumes dataset is preprocessed
        # We'll iterate through, call the decode function to get predictions
        # We'll then use regular accuracy metrics on this prediction
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        f1 = 0.
        em = 0.
        total = 0

        # If no rev_vocab, assumes it's part of the class
        if rev_vocab is None:
            rev_vocab = self.rev_vocab

        context_data, question_data, answer_data, passage_masks, question_masks = dataset
        # We assume answer_data is really a tuple
        a_t_data = answer_data[0] # Start index binary representations
        e_t_data = answer_data[1] # End index binary representations

        # ensure that the sampling size is at most the size of the dataset
        sample = min(sample, len(dataset[0]))
        # generate random sample indices into the dataset for evaluation
        sample_indices = np.random.choice(len(dataset[0]), size=sample, replace=False)

        # acquire portions of the dataset that are being sampled
        sample_contexts = context_data[sample_indices]
        sample_questions = question_data[sample_indices]
        sample_a_ts = a_t_data[sample_indices]
        sample_e_ts = e_t_data[sample_indices]

        sample_context_masks = passage_masks[sample_indices]
        sample_question_masks = question_masks[sample_indices]
        # Make it None if nothing was passed in. Else we need to get the sampled ones too.
        #Need to expand tuple, then put back into tuple form
        sample_raw_predictions_batch = (raw_predictions_batch[0][sample_indices], raw_predictions_batch[1][sample_indices]) if raw_predictions_batch is not None else None


        # run tf session to acquire predictions; predictions are in the form of index positions into the respective context paragraph
        cleaned_predictions_batch = self.answer(session, \
            test_x=(sample_context_masks, sample_questions, (sample_a_ts, sample_e_ts), sample_context_masks, sample_question_masks), \
            raw_predictions_batch = sample_raw_predictions_batch)
        logging.info("Below we print out our predicted and ground truth for later analysis. Top is predicted. Bottom is GT.")
        # compute f1 and em over the sample predictions
        for i in range(len(cleaned_predictions_batch)):
            total += 1
            sample_a_t = sample_a_ts[i] # Start index representation
            sample_e_t = sample_e_ts[i] # End index representation
            sample_context = sample_contexts[i]
            # convert sample_answer, which is a list of 0s and 1s, into index positions into the sample_context to match the predictions

            true_answer_indices = []
            start_index = int(sample_a_t)
            end_index = int(sample_e_t)
            true_answer_indices = range(start_index, end_index+1) # Get list of indices between them

            predicted_answer_indices = cleaned_predictions_batch[i]
            # convert predicted answers into a single answer string
            predicted_answer = " ".join([rev_vocab[sample_context[i]] for i in predicted_answer_indices])
            # convert true answers into a single answer string
            true_answer = " ".join([rev_vocab[sample_context[i]] for i in true_answer_indices])
            if log:
                # Log answers to have handy for future analysis and inspection
                logging.info("Predicted:")
                logging.info(predicted_answer)
                logging.info("Ground Truth:")
                logging.info(true_answer)
            # compute f1 and em
            f1 = f1 + f1_score(predicted_answer, true_answer)
            em = em + exact_match_score(predicted_answer, true_answer)

        em = 100.0 * em / total
        f1 = 100.0 * f1 / total

        # calls answer
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em


    def run_epoch(self, session, train_dataset, val_dataset, train_dir, epoch):

        # We assume dataset = (passages, questions, answers, passage_masks, question_masks)
        # We also assume each of these is a multi-dimensional numpy array
        # We assume answers is a tuple of two matrices
        passages, questions, answers, passage_masks, question_masks = train_dataset

        batch_size = self.batch_size
        a_t, e_t = answers # Expand out answers to the start and end index representations

        idxs = np.arange(0, len(passages))
        np.random.shuffle(idxs) # Shuffle them randomly

        # Get the random reordering
        shuf_passages = passages[idxs]
        shuf_questions = questions[idxs]
        shuf_a_t = a_t[idxs]
        shuf_e_t = e_t[idxs]
        shuf_passage_masks = passage_masks[idxs]
        shuf_question_masks = question_masks[idxs]

        shuf_dataset = (shuf_passages, shuf_questions, (shuf_a_t, shuf_e_t),  \
                           shuf_passage_masks, shuf_question_masks)

        # Calculate how many batches to store for eval purposes
        num_batches_to_store = int(math.ceil(self.eval_num_samples / float(batch_size)))
        logging.info("Num batches to store")
        logging.info(num_batches_to_store)
        num_store_count = 0 # How many we've stored so far

        # Need to create two. One for start indices, one for end
        stored_mini_batch_outputs_a_t = np.array([])
        stored_mini_batch_outputs_e_t = np.array([])
        batches_completed = 0

        # Iterate over the batches
        for batch_idx in range(0, len(passages), batch_size):
            logging.info("Running batch [%d/%d]" % (batch_idx, len(passages)))

            passages_batch = shuf_passages[batch_idx:batch_idx+batch_size]
            questions_batch = shuf_questions[batch_idx:batch_idx+batch_size]
            shuf_a_t_batch = shuf_a_t[batch_idx:batch_idx+batch_size]
            shuf_e_t_batch = shuf_e_t[batch_idx:batch_idx+batch_size]

            passage_masks_batch = shuf_passage_masks[batch_idx:batch_idx+batch_size]
            question_masks_batch = shuf_question_masks[batch_idx:batch_idx+batch_size]

            # Run mini-batch learning
            tic = time.time()
            mini_batch_loss, mini_batch_outputs, mini_batch_grad_norm, mini_batch_new_grad_norm = self.optimize(session, \
                (passages_batch, questions_batch, passage_masks_batch, question_masks_batch), \
                (shuf_a_t_batch, shuf_e_t_batch))
            toc = time.time()

            logging.info("Trained on batch in %f seconds." % (toc - tic))
            logging.info("Mini-batch loss: %f" % mini_batch_loss)
            logging.info("Mini-batch gradient norm: %f" % mini_batch_grad_norm)
            logging.info("Mini-batch clipped gradient norm: %f" % mini_batch_new_grad_norm)

            if num_store_count < num_batches_to_store:
                stored_mini_batch_outputs_a_t = mini_batch_outputs[0] if num_store_count == 0 else \
                    np.concatenate((stored_mini_batch_outputs_a_t, mini_batch_outputs[0]), axis=0)
                stored_mini_batch_outputs_e_t = mini_batch_outputs[1] if num_store_count == 0 else \
                    np.concatenate((stored_mini_batch_outputs_e_t, mini_batch_outputs[1]), axis=0)

                num_store_count += 1

            batches_completed += 1

            # Every save_and_evaluate_batch_num batches, we save a checkpoint of the model
            # and print a basic run over a random sample of the validation set
            if batches_completed % self.val_num_batches == 0:
                logging.info("Evaluating validation set....")
                self.sample_validation(session, val_dataset, self.val_cost_frac, self.eval_num_samples)

        end_store_index = num_batches_to_store*batch_size
        stored_mini_batch_dataset = tuple([shuf_data[:end_store_index] for shuf_data in shuf_dataset])

        # We now evaluate our F1 and EM on the last mini-batch-outputs
        # We stored the first X mini-batch outputs so we don't have to calculate again
        # We put the start and end outputs back into a tuple
        logging.info("Evaluating trained sample for F1 and EM")
        train_f1, train_em = self.evaluate_answer(session, stored_mini_batch_dataset, \
            raw_predictions_batch= (np.array(stored_mini_batch_outputs_a_t), np.array(stored_mini_batch_outputs_e_t)), \
            sample=self.eval_num_samples, log=True)

    def preprocess_dataset(self, session, dataset):
        # First we split each passage and question into arrays of ints
        # We don't do anything for questions as we assume their format is already split
        passages, questions, answers = dataset
        # We assume we have the same number of passages, questions, and answers
        for i in xrange(len(passages)):
            passages[i] = [int(word) for word in passages[i].split(" ")] # Split and convert to int
            questions[i] = [int(word) for word in questions[i].split(" ")]

        return (passages, questions, answers)

    def constrain_length_produce_mask(self, session, dataset, constrain_length=False):
        # constrain_length: If True, we constrain the length of our
        # examples by throwing out examples that are greater than our max lengths
        # We only do this in training.  If False, we pad all examples to the length
        # of the largest example in the dataset (for both questions and passages)

        # This function takes the dataset (assumed to be multi dimension numpy arrays)
        # and constrains the questions and answers to be a certain length
        # if it is less than, we pad
        # TODO: Question: What do we do for things greater than the prescribed max length?
        # How do we deal if answers are past this prescribed length?

        # We assume <Pad> is index 0 in the vocabulary, so we put it as 0 here
        zero_vector = 0

        passages, questions, answers = dataset

        if constrain_length:
            max_len_passage = self.max_length_passage
            max_len_question = self.max_length_question
        else:
            max_len_passage = max([len(passage) for passage in passages])
            max_len_question = max([len(question) for question in questions])

        question_masks = np.empty(shape=(len(passages), max_len_question))
        passage_masks = np.empty(shape=(len(passages), max_len_passage))

        # Indices we want to keep
        # If eiter passages or questions are over the max, we throw out if constrain_length is true
        keep_indices = []

        for i in xrange(len(passages)):
            # Check if within length for training purposes
            if len(passages[i]) <= max_len_passage and len(questions[i]) <= max_len_question:
                keep_indices.append(i)
            # Cut the to max length
            passages[i] = passages[i][:max_len_passage]
            questions[i] = questions[i][:max_len_question]

            # TODO: handle answers
            needed_passage = (max_len_passage - len(passages[i]))
            needed_question = (max_len_question - len(questions[i]))

            # Produce masks
            question_mask = [True]*len(questions[i]) + [False]*needed_question
            passage_mask = [True]*len(passages[i]) + [False]*needed_passage

            if len(passages[i]) < max_len_passage:
                passages[i] = passages[i] + [zero_vector]*needed_passage
            if len(questions[i]) < max_len_question:
                questions[i] = questions[i] + [zero_vector]*needed_question
            # Assign masks now
            question_masks[i, :] = question_mask
            passage_masks[i, :] = passage_mask

        # If constrain_length, restrict to those that pass both constraints
        passages = np.array(passages)[keep_indices,:] if constrain_length else np.array(passages)
        questions = np.array(questions)[keep_indices,:] if constrain_length else np.array(questions)
        answers = np.array(answers)[keep_indices,:] if constrain_length else np.array(answers)
        if constrain_length:
            passage_masks = passage_masks[keep_indices, :]
            question_masks = question_masks[keep_indices, :]

        # Check all lengths line up
        assert len(passages) == len(questions) and len(questions) == len(answers)
        assert len(answers) == len(question_masks) and len(question_masks) == len(passage_masks)
        # Return all
        return (passages, questions, answers, passage_masks, question_masks)

    # Takes answers that are represented as start end indices and turns them into the length of the passage
    # Each word in passage is either 1 or 0 (1 if in the answer, 0 otherwise)
    def expand_answers_for_model(self, session, dataset):
        # We assume dataset = (passages, questions, answers, passage_masks, question_masks)
        # We also assume each of these is a multi-dimensional numpy array
        passages, questions, answers, passage_masks, question_masks = dataset

        # TODO: Question: does right index contain the answer or not? Are these 0 indexed or 1 indexed?

        expanded_answers = np.zeros(shape=passages.shape)
        for i in xrange(len(passages)):
            expanded_answers[i, answers[i][0]:(answers[i][1]+1)] = 1
        del answers # Delete to conserve memory
        return (passages, questions, expanded_answers, passage_masks, question_masks)

    # Takes answers that are represented as start end indices and turns them into the length of the passage
    # We create two lists. One list for binary start indices representations. The other for end indices
    # We return a tuple of these lists along with the rest of the dataset
    def expand_answers_for_a_t_e_t(self, session, dataset):
        # We assume dataset = (passages, questions, answers, passage_masks, question_masks)
        # We also assume each of these is a multi-dimensional numpy array
        passages, questions, answers, passage_masks, question_masks = dataset
        # Go through and create the two new representations
        a_t_answers = np.zeros(shape=(passages.shape[0]))
        e_t_answers = np.zeros(shape=(passages.shape[0]))
        for i in xrange(len(passages)):
            a_t_answers[i] = answers[i][0]
            e_t_answers[i] = answers[i][1]

        del answers # Delete to conserve memory
        # Put into tuple
        expanded_answers = (a_t_answers, e_t_answers)
        # Return all with answers replaced by the created tuple
        return (passages, questions, expanded_answers, passage_masks, question_masks)


    def preprocess_all(self, session, dataset, constrain_length=True):
        # First we preprocess the data given
        # Convert each question + passage which is a string of space sepearated indices into a list of these indices
        dataset = self.preprocess_dataset(session, dataset)
        # For each question and answer we pad them to make them the max lengths
        # We also produce boolean masks for each
        dataset = self.constrain_length_produce_mask(session, dataset, constrain_length)
        # For each answer, we produce two binary lists. One represents binary indicator for if a word is the
        # start index or not. The other list is the same but for the end index.
        # Answers is now a tuple of these two lists
        dataset = self.expand_answers_for_a_t_e_t(session, dataset)
        return dataset

    def sample_validation(self, session, val_dataset, val_frac, eval_num_samples):
        logging.info("Running on validation set...")
        # First decide on how much data to use for validation
        num_to_keep = int(math.floor(val_frac*len(val_dataset[0])))
        logging.info(str(num_to_keep) + " validation examples being looked at")
        # Now randomly sample from these to choose
        randomly_chosen = np.random.choice(len(val_dataset[0]), num_to_keep, replace=False)
        random_val_dataset = (val_dataset[0][randomly_chosen], \
            val_dataset[1][randomly_chosen], \
            (val_dataset[2][0][randomly_chosen], val_dataset[2][1][randomly_chosen]), \
            val_dataset[3][randomly_chosen], \
            val_dataset[4][randomly_chosen])
        # First get predictions and CE Loss
        validation_loss, validation_preds = self.validate(session, random_val_dataset)
        logging.info("Validation loss: " + str(session.run(validation_loss)))
        # Now do F1 and EM sampling
        logging.info("Evaluating validation sample for F1 and EM")
        val_f1, val_em = self.evaluate_answer(session, val_dataset, \
            sample=eval_num_samples, log=True)

    def train(self, session, train_dataset, val_dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        logging.info("Number of training examples " + str(len(train_dataset[0])))
        logging.info("Number of total validation examples " + str(len(val_dataset[0])))

        # Each epoch we train on all of the data
        for epoch in range(self.start_epoch, self.epochs):
            logging.info("Running epoch " + str(epoch))
            self.run_epoch(session, train_dataset, val_dataset, train_dir, epoch)
            # TODO: add in early stopping
            logging.info("Saving model")

            self.saver.save(session, self.train_dir + '/' + self.saved_name + '_epoch' +
                str(epoch), global_step=self.global_step)

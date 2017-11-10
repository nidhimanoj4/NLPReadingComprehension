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

    def encode(self, inputs, sequence_lengths, masks=None, encoder_state_input=None):
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
        #   run bid-rectional LSTM over the passage. Concatenate forward and backward vectors at each word/time-step
        #   run bid-rectional LSTM over the question. Concatenate forward and backward vectors for the last word
        #   for each time-step in passage, concatenate state vector with the vector above

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
            p_concat_outputs = tf.concat(2, [p_outputs[0], p_outputs[1]])

        # Generate bi-lstm for question
        with vs.variable_scope("question-Bi-LSTM"):
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
            final_word_question = tf.concat(1, [q_outputs[0][:, -1,:], q_outputs[1][:, 0,:]])

        # For each word/time-step, we now concatenate with the bi-lstm representation of the last word in the associated question
        # TODO: double check this is what we want to do
        # First, we need to expand the dimension of final_word_question i.e. add a dimension in the middle for each time step
        final_word_question = tf.expand_dims(final_word_question, 1)
        # Now we multiple the middle dimension for each word in the passage
        passage_length = passages.get_shape()[1]

        max_passage_len = tf.shape(passages)[1]
        final_word_question = tf.tile(final_word_question, multiples=[1, max_passage_len, 1])

        # Now we concatenate. We want each vector for each word/time-step to get the same vector concatenated
        final_concat = tf.concat(2, [p_concat_outputs, final_word_question])

        # We return the concatenated bidirectional LSTM output for each word in the passage i.e. each time step
        # Should return (batch_size, max_length_passage, 4*hidden_size) (assuming all hidden sizes same)
        return final_concat


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep, sequence_lengths):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: a Tensor of size (batch_size, max_length_passage, knowledge_size)
        :return:
        """

        # Basic Prediction Layer
        # override output_size for now, since this is only the softmax layer
        # We assume knowledge_rep is (batch_size, max_length_passage, XXx)
        # We convert to (batch_size, max_length_passage, 2) where we output probabilities for being in the answer or not
        self.output_size = 2
        outputs = []

        passage_sequence_lengths, question_sequence_lengths = sequence_lengths

        # Run Knowledge rep through bi-directional LSTM
        with tf.variable_scope("Decode-Bi-LSTM"):
            batch_size, max_length, knowledge_size = knowledge_rep.get_shape().as_list()

            # Create forward cell
            with vs.variable_scope('forward'):
                d_lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(knowledge_size, forget_bias=1.0, state_is_tuple=True)
            # Create backward cell
            with vs.variable_scope('backward'):
                d_lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(knowledge_size, forget_bias=1.0, state_is_tuple=True)

            # Create bi-directional LSTM
            d_outputs, d_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=d_lstm_fw_cell, cell_bw = d_lstm_bw_cell, \
                inputs=knowledge_rep, dtype=tf.float64, scope="Decode-Bi-LSTM", sequence_length=passage_sequence_lengths)

            d_outputs_concat = tf.concat(2, [d_outputs[0], d_outputs[1]])

        # compute predictions as y' = sofmax(xU + b)
        with tf.variable_scope("Decode-Prediction"):

            # Create weight matrix
            U = tf.get_variable("U",
                                [knowledge_size*2, self.output_size],
                                dtype=tf.float64,
                                initializer=tf.contrib.layers.xavier_initializer())

            # Create bias vector
            b = tf.get_variable("b",
                                [1, self.output_size],
                                dtype=tf.float64,
                                initializer=tf.constant_initializer(0.0))

        max_len_passage = tf.shape(knowledge_rep)[1]

        # Since max_len_passage is dynamically computed, we cannot iterate over
        # every time step in the tensor d_outputs_concat in order to compute the prediction.
        # Instead, we reshape the tensor into a 2D matrix so that we can compute
        # predictions for all time steps with one matrix multiplication.
        # NOTE: Assuming tf.reshape unrolls dimensions in the exact opposite order
        # it rolls dimensions (initial tests on numpy seem to indicate this)
        d_outputs_reshaped = tf.reshape(d_outputs_concat, [-1, knowledge_size*2])

        outputs = tf.matmul(d_outputs_reshaped, U) + b

        # Our outputs are of size (batch_size*max_len_passage, output_size), we needed
        # to reshape them back so they are grouped by timestep
        outputs = tf.reshape(outputs, [-1, max_len_passage, self.output_size])
        return outputs

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
        self.max_length_passage = args.max_passage_length
        self.max_length_question = args.max_question_length
        self.embedding_size = args.embedding_size
        self.embed_path = args.embed_path
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        self.max_gradient_norm = args.max_gradient_norm
        self.train_dir = args.train_dir
        self.saved_name = args.saved_name
        self.eval_num_samples = args.eval_num_samples
        self.val_and_save_num_batches = args.val_and_save_num_batches
        self.val_cost_frac = args.val_cost_frac
        self.size_train_dataset = args.size_train_dataset
        self.sigma_threshold = args.sigma_threshold

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
        # 0 represents the word is not part of the answer. 1 represents it is.
        self.answers_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        # Need masks for both passages and questions
        self.mask_passage_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.mask_question_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        # Placeholders for bidirectional lstm
        # TODO: Question: is there a better way of doing this??
        # This is constant list of batch_size where each index represents the number of words in the passage
        self.passage_sequence_lengths = tf.placeholder(tf.int32, [None])
        self.question_sequence_lengths = tf.placeholder(tf.int32, [None])

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
        # ==== set up training/updating procedure ====

        # Create model saver
        self.saver = tf.train.Saver()

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
        encode_output = self.encoder.encode(encoder_inputs, sequence_lengths)

        # Assemble decoder operations on computation graph
        # We take whatever output from encoder and generate probabilities/predictions from it
        decode_output = self.decoder.decode(encode_output, sequence_lengths)

        return decode_output

    def setup_loss(self, preds):
        # This sets up the loss graph operation
        """
        Set up your loss computation here
        :return:
        """
        # TODO: Question: would it be faster to simply sum over the log of the indices where we have the correct answer?
        with vs.variable_scope("loss"):
            # Add in loss
            cross = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=self.answers_placeholder), self.mask_passage_placeholder)
            loss = tf.reduce_mean(cross)
            return loss

    def setup_learning(self, loss):
        # Think function links the loss to updating the parameters
        # We add gradient clipping because LSTMS can blow up
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

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
            self.answers_placeholder : train_y

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
            self.answers_placeholder : answers_batch
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
        # assuming output of [batch_size, self.max_length_passage, 2]; these are logits
        if raw_predictions_batch is None:
            logging.info("No predictions given so we have to calculate")
            raw_predictions_batch = self.decode(session, test_x)

        # convert logits to probabilities via softmax
        predictions_batch = tf.nn.softmax(raw_predictions_batch).eval()
        cleaned_predictions_batch = []
        contexts, _, _, context_masks, _ = test_x

        batch_size, max_length_passage = contexts.shape

        # Taking a "Maximum Subarray" approach by computing a scores Tensor
        # of size (batch_size, max_length_passage) such that
        #       scores[i, j] = predictions[i, j, 1] - predictions[i, j, 0]
        #  meaning words in which the predictions output believes it is more
        # likely to not be in the answer than otherwise will have a negative score
        # and vice versa
        posScalingTensor = tf.ones([batch_size, max_length_passage, 1], tf.float64)
        negScalingTensor = posScalingTensor * -1
        scalingTensor = tf.concat(2, [posScalingTensor, negScalingTensor])

        scores_batch = session.run(tf.reduce_sum(tf.multiply(predictions_batch, scalingTensor), 2))
        sequence_lengths =  session.run(tf.reduce_sum(tf.cast(context_masks, tf.int32), 1))

        max_scores = np.zeros([batch_size, 1])
        min_scores = np.zeros([batch_size, 1])
        for i, scores in enumerate(scores_batch):
            min_scores[i] = min(scores[:sequence_lengths[i]])
            max_scores[i] = max(scores[:sequence_lengths[i]])

        # Since the classifier tends to heavily skew towards predicting words
        # as not being in the answer, scores will be composed entirely of negative
        # numbers.  In order to compute which contiguous set of these has the
        # highest likelihood of being in the answer, we apply a bias threshold
        # sigma such that, if sigma = 0.5, then scores will be shifted such that
        # the midpoint between the highest value in scores and the lowest value in
        # scores will be 0.
        scores_ranges = max_scores - min_scores
        sigma = self.sigma_threshold
        thresholded_scores = scores_batch + -1*sigma*scores_ranges - min_scores

        # 0 - (min_score + x) = threshold*range
        # We implement Kadane's algorithm to determine the maximum value contiguous
        # subarray.
        answer_indices = []
        for i, scores in enumerate(thresholded_scores):
            max_ending_here = max_so_far = scores[0]
            curr_start_index = 0
            max_start_index = 0
            max_end_index = 0
            for j in range(1, int(sequence_lengths[i])):
                score = scores[j]
                if score > max_ending_here + score:
                    curr_start_index = j
                    max_ending_here = score
                else:
                    max_ending_here += score
                if max_ending_here > max_so_far:
                    max_so_far = max_ending_here
                    max_start_index = curr_start_index
                    max_end_index = j

            answer_indices.append(range(max_start_index, max_end_index+1))

        return answer_indices

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

        losses = []
        predictions = []

        batch_size = self.batch_size

        for batch_idx in range(0, len(passages), batch_size):
            passages_batch = passages[batch_idx:batch_idx+batch_size]
            questions_batch = questions[batch_idx:batch_idx+batch_size]
            answers_batch = answers[batch_idx:batch_idx+batch_size]
            passage_masks_batch = passage_masks[batch_idx:batch_idx+batch_size]
            question_masks_batch = question_masks[batch_idx:batch_idx+batch_size]

            loss, preds = self.test(sess, \
                (passages_batch, questions_batch, answers_batch, \
                    passage_masks_batch, question_masks_batch))

            losses.append(loss)
            predictions.append(preds)

        loss = sum(losses) / len(losses)
        preds = tf.concat(0, predictions)
        return loss, preds

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

        # ensure that the sampling size is at most the size of the dataset
        sample = min(sample, len(dataset[0]))
        # generate random sample indices into the dataset for evaluation
        sample_indices = np.random.choice(len(dataset[0]), size=sample, replace=False)

        # acquire portions of the dataset that are being sampled
        sample_contexts = context_data[sample_indices]
        sample_questions = question_data[sample_indices]
        sample_answers = answer_data[sample_indices]
        sample_context_masks = passage_masks[sample_indices]
        sample_question_masks = question_masks[sample_indices]
        # Make it None if nothing was passed in. Else we need to get the sampled ones too
        sample_raw_predictions_batch = raw_predictions_batch[sample_indices] if raw_predictions_batch is not None else None

        # run tf session to acquire predictions; predictions are in the form of index positions into the respective context paragraph
        cleaned_predictions_batch = self.answer(session, \
            test_x=(sample_contexts, sample_questions, sample_answers, sample_context_masks, sample_question_masks), \
            raw_predictions_batch = sample_raw_predictions_batch)
        logging.info("Below we print out our predicted and ground truth for later analysis. Top is predicted. Bottom is GT.")
        # compute f1 and em over the sample predictions
        for i in range(len(cleaned_predictions_batch)):
            total += 1
            sample_answer = sample_answers[i]
            sample_context = sample_contexts[i]
            # convert sample_answer, which is a list of 0s and 1s, into index positions into the sample_context to match the predictions

            true_answer_indices = []
            for j in range(len(sample_answer)):
                if sample_answer[j] == 1: true_answer_indices.append(j)

            # convert predicted answers into a single answer string
            predicted_answer = " ".join([rev_vocab[sample_context[i]] for i in cleaned_predictions_batch[i]])
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
        passages, questions, answers, passage_masks, question_masks = train_dataset

        global_step = session.run(self.global_step)

        # First we need to create batches for the data
        # Shuffle data randomly for when we batch
        # TODO: move this to tensor flow??

        if global_step == 0:
            idxs = np.arange(0, len(passages))
            np.random.shuffle(idxs) # Shuffle them randomly
            session.run(self.idxs.assign(idxs))
        else:
            idxs = session.run(self.idxs)

        batch_size = self.batch_size
        shuf_passages = passages[idxs]
        shuf_questions = questions[idxs]
        shuf_answers = answers[idxs]
        shuf_passage_masks = passage_masks[idxs]
        shuf_question_masks = question_masks[idxs]

        shuf_dataset = (shuf_passages, shuf_questions, shuf_answers,  \
                            shuf_passage_masks, shuf_question_masks)

        # Calculate how many batches to store for eval purposes
        num_batches_to_store = int(math.ceil(self.eval_num_samples / float(batch_size)))
        logging.info("Num batches to store")
        logging.info(num_batches_to_store)
        num_store_count = 0 # How many we've stored so far
        stored_mini_batch_outputs = np.array([])

        start_index = global_step*batch_size

        # Iterate over the batches
        for batch_idx in range(start_index, len(passages), batch_size):
            logging.info("Running batch [%d/%d]" % (batch_idx, len(passages)))

            passages_batch = shuf_passages[batch_idx:batch_idx+batch_size]
            questions_batch = shuf_questions[batch_idx:batch_idx+batch_size]
            answers_batch = shuf_answers[batch_idx:batch_idx+batch_size]
            passage_masks_batch = shuf_passage_masks[batch_idx:batch_idx+batch_size]
            question_masks_batch = shuf_question_masks[batch_idx:batch_idx+batch_size]

            # Run mini-batch learning
            tic = time.time()
            mini_batch_loss, mini_batch_outputs, mini_batch_grad_norm, mini_batch_new_grad_norm = self.optimize(session, \
                (passages_batch, questions_batch, passage_masks_batch, question_masks_batch), \
                answers_batch)
            toc = time.time()

            logging.info("Trained on batch in %f seconds." % (toc - tic))
            logging.info("Mini-batch loss: %f" % mini_batch_loss)
            logging.info("Mini-batch gradient norm: %f" % mini_batch_grad_norm)
            logging.info("Mini-batch clipped gradient norm: %f" % mini_batch_new_grad_norm)

            if num_store_count < num_batches_to_store:
                stored_mini_batch_outputs = mini_batch_outputs if num_store_count == 0 else \
                    np.concatenate((stored_mini_batch_outputs, mini_batch_outputs), axis=0)
                num_store_count += 1

            global_step += 1

            # Every save_and_evaluate_batch_num batches, we save a checkpoint of the model
            # and print a basic run over a random sample of the validation set
            if global_step % self.val_and_save_num_batches == 0:
                self.saver.save(session, self.train_dir + '/' + self.saved_name + '_epoch' +
                    str(epoch), global_step=batch_size*global_step)
                self.sample_validation(session, val_dataset, self.val_cost_frac, self.eval_num_samples)

        end_store_index = start_index + num_batches_to_store*batch_size
        stored_mini_batch_dataset = tuple([shuf_data[start_index:end_store_index] for shuf_data in shuf_dataset])

        # We now evaluate our F1 and EM on the last mini-batch-outputs
        # We stored the first X mini-batch outputs so we don't have to calculate again
        stored_mini_batch_outputs = np.array(stored_mini_batch_outputs) # Turn to np array
        logging.info("Evaluating trained sample for F1 and EM")
        train_f1, train_em = self.evaluate_answer(session, stored_mini_batch_dataset, \
            raw_predictions_batch=stored_mini_batch_outputs, \
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

    def preprocess_all(self, session, dataset, constrain_length=True):
        # First we preprocess the data given
        # Convert each question + passage which is a string of space sepearated indices into a list of these indices
        dataset = self.preprocess_dataset(session, dataset)
        # For each question and answer we pad them to make them the max lengths
        # We also produce boolean masks for each
        dataset = self.constrain_length_produce_mask(session, dataset, constrain_length)
        # For each answer, we produce binary values for each word in the passage
        # 0 means the word is not part of the answer, 1 means it does
        dataset = self.expand_answers_for_model(session, dataset)
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
            val_dataset[2][randomly_chosen], \
            val_dataset[3][randomly_chosen], \
            val_dataset[4][randomly_chosen])
        # First get predictions and CE Loss
        validation_loss, validation_preds = self.validate(session, random_val_dataset)
        logging.info("Validation loss: " + str(validation_loss))
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
            logging.info("Saving model")

            # Reset global step to reflect new epoch
            session.run(self.global_step.assign(0))

            self.saver.save(session, self.train_dir + '/' + self.saved_name + '_epoch' +
                str(epoch), global_step=0)

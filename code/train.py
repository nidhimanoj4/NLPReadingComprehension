from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random

import tensorflow as tf

from qa_coattention_model_final import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging
from evaluate import evaluate

# NOTE: Added libraries below
import sys

logging.basicConfig(level=logging.INFO)

# Where we specify directories and such
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "tttttksyoootrains", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "tttttksyoootrains", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "./data/squad/glove.trimmed.100.npz", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("saved_name", "model", "base name for our model")

# Where we specify data hyperparameters
tf.app.flags.DEFINE_integer("max_passage_length", 324, "Length of each passage that we require")
tf.app.flags.DEFINE_integer("max_question_length", 23, "Length of each question that we require")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("num_of_val_entries", 204, "Number of entries we want in the val dataset.")
tf.app.flags.DEFINE_integer("num_of_test_entries", 204, "Number of entries we want in the test dataset.")
tf.app.flags.DEFINE_integer("num_of_train_entries", 3877, "Number of entries we want in the train dataset.")

# Where we specify training hyper parameters
tf.app.flags.DEFINE_float("start_learning_rate", 0.01, "Learning rate we start with before decaying.")
tf.app.flags.DEFINE_float("learning_decay_rate", 0.96, "Rate at which learning rate decays exponentially.")
tf.app.flags.DEFINE_integer("num_decay_steps", 1000, "decayed_learning_rate = learning_rate*decay_rate ^ (global_step / num_decay_steps)")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")

tf.app.flags.DEFINE_boolean('should_use_new_loss', False, 'If should use our new loss function.')
tf.app.flags.DEFINE_boolean('should_use_dp_prediction', False, 'If should use our DP for answer prediction.')


# Temp hack for baseline
tf.app.flags.DEFINE_integer("start_epoch", 0, "Epoch to start with.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("size_train_dataset", 20, "The size of the training dataset")

# Where we specify model architecture hyperparameters
tf.app.flags.DEFINE_float("dropout", .85, "Fraction of units randomly kept on non-recurrent connections.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each layer in the Encoder.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("eval_num_samples", 100, "How many samples to evaluate.")
tf.app.flags.DEFINE_integer("val_num_batches", 50, "Per how many batches do we run on a validation sample and save the model.")
tf.app.flags.DEFINE_integer("num_keep_checkpoints", 5, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_float("val_cost_frac", 0.05, "Fraction of validation set used for periodic evaluation.")
tf.app.flags.DEFINE_float("sigma_threshold", 0.5, "Threshold to apply to answer probabilities in order to determine answer indices")
tf.app.flags.DEFINE_float("l2_lambda", 0.01, "Amount of L2 regularization we want to apply to our parameters.")

# Where we specify model architecture add-ons
tf.app.flags.DEFINE_boolean("quadratic_form", False, "Whether to convert coattention to a quadratic form by adding a new weight matrix.")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    print("CKPT")
    print(ckpt)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print(ckpt.model_checkpoint_path)
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


#NOTE: Added below to process data

def load_token_file(file_name):
    data = []
    file_contents = open(file_name, "rb")
    for line in file_contents:
        # Assumes already in the space delimited token index format
        data.append(line.rstrip()) # Get rid of trailing newline
    return data

def load_span_file(file_name):
    data = []
    file_contents = open(file_name, "rb")
    for line in file_contents:
        # Assumes space delimited
        left_point, right_point = line.rstrip().split(" ")
        # Convert to ints and add to list
        data.append((int(left_point), int(right_point)))
    return data
    return True

def load_datasets():
    # Do what you need to load datasets from FLAGS.data_dir
    # We load the .ids. file because in qa_answer they are also loaded
    dataset_dir = FLAGS.data_dir
    abs_dataset_dir = os.path.abspath(dataset_dir)
    # We will no longer use the train dataset, we are only going to be using the val dataset and splitting it up into our own test, train, and val datasets
    # NOTE: get all the files that we want to load
    answer_file = os.path.join(abs_dataset_dir, "val.answer")
    context_file = os.path.join(abs_dataset_dir, "val.context")
    question_file = os.path.join(abs_dataset_dir, "val.question")
    ids_context_file = os.path.join(abs_dataset_dir, "val.ids.context")
    ids_question_file = os.path.join(abs_dataset_dir, "val.ids.question")
    span_answer_file = os.path.join(abs_dataset_dir, "val.span")

    # NOTE: Get data by loading in the files we just made using load_token_file
    # For context and question, we assume each item in the list is a string
    # The string is space seperated list of tokens that correspond to indices in the vocabulary
    # We assume this since this is what's passed in for qa_answer.py
    # Since this isn't in qa_answer.py, we assume each item in the list to be a tuple
    # The first place in the tuple is the starting index relative to the passage
    # NOTE: it's possible for both values to be the same
    valid_context_data = load_token_file(ids_context_file)
    valid_question_data = load_token_file(ids_question_file)
    valid_answer_data = load_span_file(span_answer_file)

    if (len(valid_context_data) != len(valid_question_data) or len(valid_context_data) != len(valid_answer_data)):
        print('Error: the number of paragraphs, questions, and answers do not match')
          
    # Make an array of indices 0 ... (len(valid_context_data) - 1)
    indices_available = []
    for index in range(0, len(valid_context_data)):
        indices_available.append(index)
    
    new_val_context_data = []
    new_val_question_data = []
    new_val_answer_data = []

    new_test_context_data = []
    new_test_question_data = []
    new_test_answer_data = []
    
    new_train_context_data = []
    new_train_question_data = []
    new_train_answer_data = []
    
    # Note: set up the new_val datasets for context, question, answer
    for i in range(FLAGS.num_of_val_entries):
        rand_index = indices_available[random.randrange(0,len(indices_available))]
        new_val_context_data.append(valid_context_data[rand_index])
        new_val_question_data.append(valid_question_data[rand_index])
        new_val_answer_data.append(valid_answer_data[rand_index])
        indices_available.remove(rand_index)
    for i in range(FLAGS.num_of_test_entries):
        rand_index = indices_available[random.randrange(0,len(indices_available))]
        new_test_context_data.append(valid_context_data[rand_index])
        new_test_question_data.append(valid_question_data[rand_index])
        new_test_answer_data.append(valid_answer_data[rand_index])
        indices_available.remove(rand_index)
    for index in indices_available:
        new_train_context_data.append(valid_context_data[index])
        new_train_question_data.append(valid_question_data[index])
        new_train_answer_data.append(valid_answer_data[index])
              
    print('Length of val dataset = ', len(new_val_context_data))
    print('Length of test dataset = ', len(new_test_context_data))
    print('Length of train dataset = ', len(new_train_context_data))

    # Merge data
    new_val_dataset = (new_val_context_data, new_val_question_data, new_val_answer_data)
    new_test_dataset = (new_test_context_data, new_test_question_data, new_test_answer_data)
    new_train_dataset = (new_train_context_data, new_train_question_data, new_train_answer_data)
    return (new_val_dataset, new_test_dataset, new_train_dataset)

#def printAvgParagraphLength(valid_context_data):
#    number_of_paragraphs_in_context = len(valid_context_data)
#    if number_of_paragraphs_in_context == 0:
#        return 0
#
#    total_sum_of_word_ids_in_all_paragraphs = 0
#    for paragraph in valid_context_data:
#        arr_of_word_ids_in_paragraph = paragraph.split()
#        if (total_sum_of_word_ids_in_all_paragraphs == 0):
#            print('Paragraph = ', paragraph, '\n')
#            print('Number of characters with spaces in paragraph = ', len(paragraph))
#            print('Number of words in paragraph = ', len(arr_of_word_ids_in_paragraph))
#        total_sum_of_word_ids_in_all_paragraphs += len(arr_of_word_ids_in_paragraph)
#
#    print('total_sum_of_word_ids_in_all_paragraphs = ', total_sum_of_word_ids_in_all_paragraphs)
#    avg_num_of_word_ids_in_paragraphs = (total_sum_of_word_ids_in_all_paragraphs * 1.0) / number_of_paragraphs_in_context
#    print('number_of_paragraphs_in_context = ', number_of_paragraphs_in_context, '\n')
#    return avg_num_of_word_ids_in_paragraphs

def printAvgLength(valid_data):
    number_of_phrases = len(valid_data)
    if number_of_phrases == 0:
        return 0
    total_sum_of_word_ids_in_all_phrases = 0
    for phrase in valid_data:
        total_sum_of_word_ids_in_all_phrases += len(phrase.split())
    avg_num_of_word_ids_in_phrases = (total_sum_of_word_ids_in_all_phrases * 1.0) / number_of_phrases
    return avg_num_of_word_ids_in_phrases

def printAvgAnswerLength(valid_data):
    number_of_answers = len(valid_data)
    if number_of_answers == 0:
        return 0
    total_num_words_in_all_answers = 0
    for start_index, end_index in valid_data:
        length_of_answer = end_index - start_index + 1
        total_num_words_in_all_answers += length_of_answer
    avg_num_of_words_in_answers = (total_num_words_in_all_answers * 1.0) / number_of_answers
    return avg_num_of_words_in_answers

def main(_):
    val_dataset, test_dataset, train_dataset = load_datasets()

    val_context_data, val_question_data, val_answer_data = val_dataset
    #avg_num_of_word_ids_in_paragraphs = printAvgParagraphLength(valid_context_data)
    val_avg_num_of_words_in_paragraphs = printAvgLength(val_context_data)
    val_avg_num_of_words_in_questions = printAvgLength(val_question_data)
    val_avg_num_of_words_in_answers = printAvgAnswerLength(val_answer_data)
    print('val_avg_num_of_words_in_paragraphs = ', val_avg_num_of_words_in_paragraphs, '\n',     'val_avg_num_of_words_in_questions = ', val_avg_num_of_words_in_questions, '\n', 'val_avg_num_of_words_in_answers = ', val_avg_num_of_words_in_answers, '\n')
    
    test_context_data, test_question_data, test_answer_data = test_dataset
    test_avg_num_of_words_in_paragraphs = printAvgLength(test_context_data)
    test_avg_num_of_words_in_questions = printAvgLength(test_question_data)
    test_avg_num_of_words_in_answers = printAvgAnswerLength(test_answer_data)
    print('test_avg_num_of_words_in_paragraphs = ', test_avg_num_of_words_in_paragraphs, '\n',     'test_avg_num_of_words_in_questions = ', test_avg_num_of_words_in_questions, '\n', 'test_avg_num_of_words_in_answers = ', test_avg_num_of_words_in_answers, '\n')

    train_context_data, train_question_data, train_answer_data = train_dataset
    train_avg_num_of_words_in_paragraphs = printAvgLength(train_context_data)
    train_avg_num_of_words_in_questions = printAvgLength(train_question_data)
    train_avg_num_of_words_in_answers = printAvgAnswerLength(train_answer_data)
    print('train_avg_num_of_words_in_paragraphs = ', train_avg_num_of_words_in_paragraphs, '\n',     'train_avg_num_of_words_in_questions = ', train_avg_num_of_words_in_questions, '\n', 'train_avg_num_of_words_in_answers = ', train_avg_num_of_words_in_answers, '\n')
    
    print('Hello')


#    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
#    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
#    vocab, rev_vocab = initialize_vocab(vocab_path)
#
#    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size, quadratic_form=FLAGS.quadratic_form)
#    decoder = Decoder(output_size=FLAGS.output_size)
#
#    qa = QASystem(encoder, decoder, rev_vocab, FLAGS)
#
#    if not os.path.exists(FLAGS.log_dir):
#        os.makedirs(FLAGS.log_dir)
#    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
#    logging.getLogger().addHandler(file_handler)
#
#    print(vars(FLAGS))
#    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
#        json.dump(FLAGS.__flags, fout)
#
#    with tf.Session() as sess:
#        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
#        initialize_model(sess, qa, load_train_dir)
#
#        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
#
#        train_dataset = qa.preprocess_all(sess, train_dataset)
#        valid_dataset = qa.preprocess_all(sess, valid_dataset, constrain_length=False)
#        qa.train(sess, train_dataset, valid_dataset, save_train_dir)

if __name__ == "__main__":
    tf.app.run()

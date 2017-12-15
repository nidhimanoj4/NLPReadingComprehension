# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random
import time
import logging
import collections

import numpy as np
from six.moves import xrange
from os.path import join as pjoin
import tensorflow as tf

from evaluate import exact_match_score, f1_score, evaluate
import sys
import math
import re

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
    

def load_datasets():
    # Do what you need to load datasets from FLAGS.data_dir
    # We load the .ids. file because in qa_answer they are also loaded
    dataset_dir = FLAGS.data_dir
    abs_dataset_dir = os.path.abspath(dataset_dir)
    # We will no longer use the train dataset, we are only going to be using the val dataset and splitting it up into our own test, train, and val datasets
    # NOTE: get all the files that we want to load
    train_answer_file = os.path.join(abs_dataset_dir, "new_train_answer_data")
    train_answer_span_file = os.path.join(abs_dataset_dir, "new_train_answer_span_data")
    train_context_file = os.path.join(abs_dataset_dir, "new_train_context_data")
    train_question_file = os.path.join(abs_dataset_dir, "new_train_question_data")
    test_answer_file = os.path.join(abs_dataset_dir, "new_test_answer_data")
    test_answer_span_file = os.path.join(abs_dataset_dir, "new_test_answer_span_data")
    test_context_file = os.path.join(abs_dataset_dir, "new_test_context_data")
    test_question_file = os.path.join(abs_dataset_dir, "new_test_question_data")
    val_answer_file = os.path.join(abs_dataset_dir, "new_val_answer_data")
    val_answer_span_file = os.path.join(abs_dataset_dir, "new_val_answer_span_data")
    val_context_file = os.path.join(abs_dataset_dir, "new_val_context_data")
    val_question_file = os.path.join(abs_dataset_dir, "new_val_question_data")
    vocab_file = os.path.join(abs_dataset_dir, "vocab.dat")
    
    demo_answer_file = os.path.join(abs_dataset_dir, "new_demo_answer_data")
    demo_context_file = os.path.join(abs_dataset_dir, "new_demo_context_data")
    demo_question_file = os.path.join(abs_dataset_dir, "new_demo_question_data")

    train_answer_data = load_token_file(train_answer_file)
    train_answer_span_data = load_span_file(train_answer_span_file)
    train_context_data = load_token_file(train_context_file)
    train_question_data = load_token_file(train_question_file)
    
    test_answer_data = load_token_file(test_answer_file)
    test_answer_span_data = load_span_file(test_answer_span_file)
    test_context_data = load_token_file(test_context_file)
    test_question_data = load_token_file(test_question_file)
    
    valid_answer_data = load_token_file(val_answer_file)
    valid_answer_span_data = load_span_file(val_answer_span_file)
    valid_context_data = load_token_file(val_context_file)
    valid_question_data = load_token_file(val_question_file)
    
    vocab_token_data = load_token_file(vocab_file)
    
    demo_answer_data = load_token_file(demo_answer_file)
    demo_context_data = load_token_file(demo_context_file)
    demo_question_data = load_token_file(demo_question_file)
    
    # Merge data
    val_dataset = (valid_context_data, valid_question_data, valid_answer_data, valid_answer_span_data)
    test_dataset = (test_context_data, test_question_data, test_answer_data, test_answer_span_data)
    train_dataset = (train_context_data, train_question_data, train_answer_data, train_answer_span_data)
    
    demo_dataset = (demo_context_data, demo_question_data, demo_answer_data)
    
    return (val_dataset, test_dataset, train_dataset, vocab_token_data, demo_dataset)

def sumVectorsOfSameDimension(vector1, scale, vector2):
    """
        Assumes vectors are of the same dimension.
        Implements vector1 += scale * vector2.
        @param vector vector1: the vector which is mutated.
        @param float scale
        @param vector vector2
        """
    if len(vector1) != len(vector2):
        return
    for index in range(len(vector1)):
        vector1[index] = vector1[index] + (scale * vector2[index])
    return vector1

def getNumWordsCommonInPhrases(phrase1, phrase2):
    # We want to return the number of words in phrase that are also in question (phrase1)
    num_common_words = 0
    words = phrase2.split(" ")
    question = re.split("\.|;|,|\(|\)|\?|\!|\:| ", phrase1)
    lower_question = []
    for a in question:
        lower_question.append(a.lower())
    keep_track = []
    #print("the phrase: ", words)
    #print("the question: ", question)
    for word in words:
        low = word.lower()
        if (low in lower_question) and (low not in keep_track) and (low != " "):
            keep_track.append(low)
            #print(low)
            num_common_words = num_common_words + 1
    return num_common_words

#Converting a word into its indices form
def convertWordToIndex(word, vocab_token_data):
    return vocab_token_data.index(word)

def evalFnOverNumWordsInCorrectAnswer(predicted_answer, correct_answer):
    num_common_words = getNumWordsCommonInPhrases(predicted_answer, correct_answer)
    words_in_correct_answer = correct_answer.split(" ")
    return (num_common_words * 1.0) / len(words_in_correct_answer)

def evalFnOverNumWordsInPredictedAnswer(predicted_answer, correct_answer):
    num_common_words = getNumWordsCommonInPhrases(predicted_answer, correct_answer)
    words_in_predicted_answer = predicted_answer.split(" ")
    return (num_common_words * 1.0) / len(words_in_predicted_answer)

def evalFnIntersectionOverUnion(predicted_answer, correct_answer):
    num_common_words = getNumWordsCommonInPhrases(predicted_answer, correct_answer)
    union = len(correct_answer.split(" ")) + len(predicted_answer.split(" ")) - num_common_words
    return (num_common_words * 1.0) / (union)

def evalFnAverage(predicted_answer, correct_answer):
    sum_eval_metrics = evalFnOverNumWordsInCorrectAnswer(predicted_answer, correct_answer) + evalFnOverNumWordsInPredictedAnswer(predicted_answer, correct_answer) + evalFnIntersectionOverUnion(predicted_answer, correct_answer)
    return sum_eval_metrics / 3.0

#Baseline predictor for one passage, question, and answer
def baselinePredictor(passage, question):
    #phrase with maximum count
    max_phrase = ""
    #Max count of matching words or cosine similarity
    max_count = 0
    #The phrases within the paragraph
    #        print('passage = ', passage)
    phrases = re.split("\.|;|,|\(|\)|\?|\!|\:", passage)
        
    # VERSION 1 (Baseline): Pick the phrase with the max number of common words to the question as the predicted answer
    for phrase in phrases:
        num_words_common_in_question = getNumWordsCommonInPhrases(question, phrase)
        #print(num_words_common_in_question)
        if num_words_common_in_question > max_count:
            #print(num_words_common_in_question)
            #print("The phrase: ", phrase)
            max_count = num_words_common_in_question
            max_phrase = phrase
    return max_phrase

#Baseline: To get the context with question and answer
def baseline(train_dataset, vocab_token_data):
    train_context = train_dataset[0]
    train_question = train_dataset[1]
    train_answer = train_dataset[2]
    #We don't need it
    train_answer_span = train_dataset[3]
    
    list_of_evaluation_metrics_over_correct_answer = []
    list_of_evaluation_metrics_over_predicted_answer = []
    list_of_evaluation_metrics_avg = []
    list_of_evaluation_metrics_intersection_over_union = []
    
    # Load the glove vectors data
    glove_trimmed_data = np.load('data/squad/glove.trimmed.100.npz')
    glove_vector_data = glove_trimmed_data['glove']
    # print('glove_vector_data = ', glove_vector_data)
    
    # Iterate through the context and questions
    for i in range(len(train_question)):
        passage = train_context[i] # text
        question = train_question[i] # text
        correct_answer = train_answer[i] # text
        correct_answer_span = train_answer_span[i] #span of two integers of answer

        predicted_answer = baselinePredictor(passage, question)
        # #phrase with maximum count
        # max_phrase = ""
        # #Max count of matching words or cosine similarity
        # max_count = 0
        # #The phrases within the paragraph
        # #        print('passage = ', passage)
        # phrases = re.split("\.|;|,|\(|\)|\?|\!|\:", passage)
        
        # # VERSION 1 (Baseline): Pick the phrase with the max number of common words to the question as the predicted answer
        # for phrase in phrases:
        #     num_words_common_in_question = getNumWordsCommonInPhrases(question, phrase)
        #     #print(num_words_common_in_question)
        #     if num_words_common_in_question > max_count:
        #         #print(num_words_common_in_question)
        #         #print("The phrase: ", phrase)
        #         max_count = num_words_common_in_question
        #         max_phrase = phrase
    
        # VERSION 2: (GLoVE) Pick the phrase with the greatest cosine similarity in glove vectors for phrase and question
        #        for phrase in phrases:
        #            # gloveCosineSimilarityValue will be between [-1, 1]
        #            gloveCosineSimilarityValue = getGloveCosineSimilarityValue(question, phrase, glove_vector_data, vocab_token_data)
        #            if gloveCosineSimilarityValue > max_count:
        #                max_count = gloveCosineSimilarityValue
        #                max_phrase = phrase
        
        # Max phrase is now the predicted answer in baseline
        #predicted_answer = max_phrase
        print('question = \"', question, '\"')
        print('predicted_answer = ', predicted_answer)
        #        print('correct_answer = ', correct_answer)
        
        evaluation_metric_over_correct_answer = evalFnOverNumWordsInCorrectAnswer(predicted_answer, correct_answer)
        list_of_evaluation_metrics_over_correct_answer.append(evaluation_metric_over_correct_answer)
        
        evaluation_metric_over_predicted_answer = evalFnOverNumWordsInPredictedAnswer(predicted_answer, correct_answer)
        list_of_evaluation_metrics_over_predicted_answer.append(evaluation_metric_over_predicted_answer)
        
        evaluation_metric_intersection_over_union = evalFnIntersectionOverUnion(predicted_answer, correct_answer)
        list_of_evaluation_metrics_intersection_over_union.append(evaluation_metric_intersection_over_union)
        
        evaluation_metric_avg = evalFnAverage(predicted_answer, correct_answer)
        list_of_evaluation_metrics_avg.append(evaluation_metric_avg)

    avg_evaluation_metric_over_correct_answer = sum(list_of_evaluation_metrics_over_correct_answer) / (len(list_of_evaluation_metrics_over_correct_answer) * 1.0)
    #print ('avg_evaluation_metric_over_correct_answer = ', avg_evaluation_metric_over_correct_answer)

    avg_evaluation_metric_over_predicted_answer = sum(list_of_evaluation_metrics_over_predicted_answer) / (len(list_of_evaluation_metrics_over_predicted_answer) * 1.0)
    #print ('avg_evaluation_metric_over_predicted_answer = ', avg_evaluation_metric_over_predicted_answer)
    
    avg_evaluation_metric_intersection_over_union = sum(list_of_evaluation_metrics_intersection_over_union) / (len(list_of_evaluation_metrics_intersection_over_union) * 1.0)
    #print ('avg_evaluation_metric_intersection_over_union = ', avg_evaluation_metric_intersection_over_union)
    
    avg_evaluation_metric_avg = sum(list_of_evaluation_metrics_avg) / (len(list_of_evaluation_metrics_avg) * 1.0)
    #print ('avg_evaluation_metric_avg = ', avg_evaluation_metric_avg)

def printAvgLength(valid_data):
    number_of_phrases = len(valid_data)
    if number_of_phrases == 0:
        return 0
    total_sum_of_word_ids_in_all_phrases = 0
    for phrase in valid_data:
        total_sum_of_word_ids_in_all_phrases += len(phrase.split(" "))
    avg_num_of_word_ids_in_phrases = (total_sum_of_word_ids_in_all_phrases * 1.0) / number_of_phrases
    return avg_num_of_word_ids_in_phrases


#def printAvgAnswerLengthWithAnswerSpans(valid_data):
#    number_of_answers = len(valid_data)
#    if number_of_answers == 0:
#        return 0
#    total_num_words_in_all_answers = 0
#    for start_index, end_index in valid_data:
#        length_of_answer = end_index - start_index + 1
#        total_num_words_in_all_answers += length_of_answer
#    avg_num_of_words_in_answers = (total_num_words_in_all_answers * 1.0) / number_of_answers
#    return avg_num_of_words_in_answers

#Get the feature extractor. Takes in the phrase and generates a feature vector
def featureExtractor(predicted_substring, question, predicted_start_index, vocab_list):
	# print('In the featureExtractor. predicted_substring: ', predicted_substring)
	features = collections.defaultdict(int)
	#Question type feature vector
	question_elems = question.split(' ')
	#Case insensitive
	question_components = []
	for elem in question_elems:
		question_components.append(elem.lower())

	#For different question types
	# if ('what' in question_components):
	# 	features['what'] = 1
	# if ('who' in question_components):
	# 	features['who'] = 1
	# if ('when' in question_components):
	# 	features['when'] = 1
	# if ('where' in question_components):
	# 	features['where'] = 1
	# if ('how' in question_components):
	# 	features['how'] = 1
	# if ('why' in question_components):
	# 	features['why'] = 1
	# if ('which' in question_components):
	# 	features['which'] = 1

	#Character length of substring
	char_length = len(predicted_substring)
	if char_length >= 20:
		features['char_length >= 20'] = 1
	# if char_length < 20:
	# 	features['char_length < 20'] = 1

	#Index start of feature
	features['start_index'] = predicted_start_index
	length_predicted_answer = len(predicted_substring.split(" "))
	features['length_predicted_answer'] = length_predicted_answer
	features['end_index'] = predicted_start_index + length_predicted_answer - 1
	features['contains .'] = ('.' in predicted_substring)
	features['contains ,'] = (',' in predicted_substring)
	features['contains !'] = ('!' in predicted_substring)
	features['contains ?'] = ('?' in predicted_substring)
	features['contains :'] = (':' in predicted_substring)
	features['contains ;'] = (';' in predicted_substring)

	# for word in vocab_list:
	# 	features['occurrences of ' + word] = 0
	for word in predicted_substring.split(" "):
		lowered_word = word.lower()
		if lowered_word in vocab_list:
			features['occurrences of ' + lowered_word] += 1
	
	features['words in common with question'] = getNumWordsCommonInPhrases(predicted_substring, question)

	num_uppercase_letters = 0
	for i in range(len(predicted_substring)):
		char_in_predicted_substring = predicted_substring[i]
		if char_in_predicted_substring.isupper():
			num_uppercase_letters += 1
	if num_uppercase_letters == 0:
		features['num_of_uppercase_letters = 0'] = 1
	if num_uppercase_letters == 1:
		features['num_of_uppercase_letters = 1'] = 1
	if num_uppercase_letters == 2:
		features['num_of_uppercase_letters = 2'] = 1
	if num_uppercase_letters >= 3:
		features['num_of_uppercase_letters >= 3'] = 1
	features['has_uppercase_letter'] = (num_uppercase_letters > 0)


	if predicted_substring != "" and predicted_substring[0].isupper():
		features['first_char_is_uppercase_letter'] = 1
	return features

#Get the dot product of two vectors
def dotProduct(vec1, vec2):
    if len(vec1) < len(vec2):
        return dotProduct(vec2, vec1)
    else:
        return sum(vec1.get(f, 0) * v for f, v in vec2.items())

#Increment the weight vector
def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


#Evaluate the predictor
def evaluatePredictor(examples, predictor):
	context_data, question_data, answer_data, answer_span_data  = examples
	# total_performance_error = 0.0
	total_intersection_over_union = 0.0
	total_over_predicted = 0.0
	total_over_correct = 0.0
	total_avg_all_evaluations = 0.0

	threshold_half = 0.20
	num_entries_with_intersection_over_union_greater_than_half = 0.0

	for index in range(len(context_data)):
		passage = context_data[index]
		question = question_data[index]
		correct_answer = answer_data[index]
		correct_answer_span_start, correct_answer_span_end = answer_span_data[index]
		predicted_start_index, predicted_substring = predictor(passage, question)

		print('correct answer: ', correct_answer)
		print('predicted answer: ', predicted_substring, '\n')
		# error_in_start_index_prediction = (predicted_start_index - correct_answer_span_start)**2
		# total_performance_error += (error_in_start_index_prediction)
		#Get the evaluation results for each of the three evaluation functions
		eval_union =  evalFnIntersectionOverUnion(predicted_substring, correct_answer)
		eval_predicted = evalFnOverNumWordsInPredictedAnswer(predicted_substring, correct_answer)
		eval_correct = evalFnOverNumWordsInCorrectAnswer(predicted_substring, correct_answer)
		eval_avg = evalFnAverage(predicted_substring, correct_answer)
		
		#Sum up the calculated evaluations
		total_intersection_over_union += eval_union
		total_over_predicted += eval_predicted
		total_over_correct += eval_correct
		total_avg_all_evaluations += eval_avg

		if eval_union >= threshold_half:
			num_entries_with_intersection_over_union_greater_than_half += 1

	# average_performance_error = total_performance_error / len(context_data)
	# print('Evaluate predictor: total_performance_error = ', total_performance_error)
	# return average_performance_error
	avg_intersection_over_union = total_intersection_over_union / len(context_data)
	print('avg_intersection_over_union: ', avg_intersection_over_union)
	avg_over_predicted = total_over_predicted / len(context_data)
	print('avg_over_predicted: ', avg_over_predicted)
	avg_over_correct = total_over_correct / len(context_data)
	print('avg_over_correct: ', avg_over_correct)
	avg_all_evaluations = total_avg_all_evaluations / len(context_data)
	print('avg_all_evaluations: ', avg_all_evaluations)

	avg_num_entries_with_intersection_over_union_greater_than_half = num_entries_with_intersection_over_union_greater_than_half / len(context_data)
	print('avg_num_entries_with_intersection_over_union_greater_than_half: ', avg_num_entries_with_intersection_over_union_greater_than_half)

	return avg_num_entries_with_intersection_over_union_greater_than_half

#The Stochastic Gradient Descent framework
def learnPredictor(trainExamples, valExamples, testExamples, numIters, eta, vocab_list, avg_correct_answer_length_in_train):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")

    #Calculate the gradient for SGD
    def getGradient(y_prediction_start_index, y_start_index, y_predicted_substring):
    	# y_prediction is the predicted start index in the passage where we think the answer should begin
    	# y is the start index of the correct answer in the passage
    	gradientLoss = {}
    	# gradientLoss += gradientLoss + 2 ( (phi_y_predict * w) - y) * (phi_y_predict)
    	feature_vector_y_prediction = featureExtractor(y_predicted_substring, question, y_prediction_start_index, vocab_list)
    	score_y_predict = dotProduct(feature_vector_y_prediction, weights)
    	score = score_y_predict - y_start_index

    	for key, value in feature_vector_y_prediction.items():
    		gradientLoss[key] = 2 * score * value

    	#increment(gradientLoss, 2 * score, feature_vector_y_prediction)
    	return gradientLoss    

    context_data, question_data, answer_data, answer_span_data = trainExamples

    def predictor(passage, question):
    	scores_for_each_sliding_substring = []
    	words_in_passage = passage.split(" ")

    	# print('passage: ', passage)
    	# print('question: ', question)
    	for index in range(len(words_in_passage)):
    		if index + avg_correct_answer_length_in_train <= len(words_in_passage):
    			sliding_substring = ""
    			for i in range(avg_correct_answer_length_in_train):
    				sliding_substring += words_in_passage[index + i] + " "

    			feature_vector_sliding_substring = featureExtractor(sliding_substring, question, index, vocab_list)
    			# print("the feature vector: ", feature_vector_sliding_substring)
    			score_for_sliding_substring = dotProduct(feature_vector_sliding_substring, weights)
    			# print("Dot product score: ", score_for_sliding_substring, '\n')
    			scores_for_each_sliding_substring.append(score_for_sliding_substring)

    			# print("sliding_substring = ", sliding_substring, " score_for_sliding_substring = ", score_for_sliding_substring)
    	# print("all scores ", scores_for_each_sliding_substring)
    	max_score_for_sliding_substring = max(scores_for_each_sliding_substring)
    	start_index_of_sliding_substring = scores_for_each_sliding_substring.index(max_score_for_sliding_substring)
    	# print("score: ", max_score_for_sliding_substring, '\n')

    	
    	# return predicted start index of the correct answer
    	predicted_sliding_substring = ""
    	for i in range(avg_correct_answer_length_in_train):
    		predicted_sliding_substring += words_in_passage[start_index_of_sliding_substring + i] + " "
    	return start_index_of_sliding_substring, predicted_sliding_substring

    for i in range(numIters):
    	for index in range(len(context_data)):
    		gradientLoss = {}
    		passage = context_data[index]
    		question = question_data[index]
    		correct_answer = answer_data[index]
    		correct_answer_span_start, correct_answer_span_end = answer_span_data[index]

    		y_predicted_start_index, y_predicted_substring = predictor(passage, question)
    		print("predicted index: ", y_predicted_start_index, " correct index: ", correct_answer_span_start)
    		print("predicted substring: ", y_predicted_substring)
    		print("correct substring: ", correct_answer)
    		print()
    		gradientLoss = getGradient(y_predicted_start_index, correct_answer_span_start, y_predicted_substring)
    		increment(weights, -1 * eta, gradientLoss)
    		#print("gradientLoss: ", gradientLoss)
    		# print("weights after iteration ", i, " and passage ", index, ": " weights)
    	print("Iteration: ", i)
    
    print('train example predictor evaluation: ', evaluatePredictor(trainExamples, predictor))
    #print('val example predictor evaluation: ', evaluatePredictor(valExamples, predictor))
    print('test example predictor evaluation: ', evaluatePredictor(testExamples, predictor))

    # END_YOUR_CODE
    return weights


def getAverageCorrectAnswerLengthInTrain(train_dataset):
	train_context_data, train_question_data, train_answer_data, train_answer_span_data = train_dataset
	total_sum_of_lengths_correct_answer = 0.0
	for index in range(len(train_answer_span_data)):
		start, end = train_answer_span_data[index]
		length = end - start + 1
		total_sum_of_lengths_correct_answer += length
	average_length_correct_answer = total_sum_of_lengths_correct_answer / len(train_answer_span_data)
	return int(math.ceil(average_length_correct_answer))


def main(_):
    print('Loading the datasets: ')
    val_dataset, test_dataset, train_dataset, vocab_token_data, demo_dataset = load_datasets()

    val_context_data, val_question_data, val_answer_data, val_answer_span_data = val_dataset
    #avg_num_of_word_ids_in_paragraphs = printAvgParagraphLength(valid_context_data)
    val_avg_num_of_words_in_paragraphs = printAvgLength(val_context_data)
    val_avg_num_of_words_in_questions = printAvgLength(val_question_data)
    val_avg_num_of_words_in_answers = printAvgLength(val_answer_data)
    #print('val_avg_num_of_words_in_paragraphs = ', val_avg_num_of_words_in_paragraphs, '\n',     'val_avg_num_of_words_in_questions = ', val_avg_num_of_words_in_questions, '\n', 'val_avg_num_of_words_in_answers = ', val_avg_num_of_words_in_answers, '\n')
    
    test_context_data, test_question_data, test_answer_data, test_answer_span_data = test_dataset
    test_avg_num_of_words_in_paragraphs = printAvgLength(test_context_data)
    test_avg_num_of_words_in_questions = printAvgLength(test_question_data)
    test_avg_num_of_words_in_answers = printAvgLength(test_answer_data)
    #print('test_avg_num_of_words_in_paragraphs = ', test_avg_num_of_words_in_paragraphs, '\n',     'test_avg_num_of_words_in_questions = ', test_avg_num_of_words_in_questions, '\n', 'test_avg_num_of_words_in_answers = ', test_avg_num_of_words_in_answers, '\n')
    
    train_context_data, train_question_data, train_answer_data, train_answer_span_data = train_dataset
    train_avg_num_of_words_in_paragraphs = printAvgLength(train_context_data)
    train_avg_num_of_words_in_questions = printAvgLength(train_question_data)
    train_avg_num_of_words_in_answers = printAvgLength(train_answer_data)
    #print('train_avg_num_of_words_in_paragraphs = ', train_avg_num_of_words_in_paragraphs, '\n',     'train_avg_num_of_words_in_questions = ', train_avg_num_of_words_in_questions, '\n', 'train_avg_num_of_words_in_answers = ', train_avg_num_of_words_in_answers, '\n')
    
    #print('\n', 'Baseline model and evaluation function on train dataset: ')
    #baseline(train_dataset, vocab_token_data)
    
    #print('\n', 'Baseline model and evaluation function on val dataset: ')
    #baseline(val_dataset, vocab_token_data)
    
    #To run demo, uncomment line
    #baseline(demo_dataset, vocab_token_data)



    average_correct_answer_length = getAverageCorrectAnswerLengthInTrain(train_dataset)

    print('Learning weights from train examples: ')
    weights = learnPredictor(trainExamples=train_dataset, valExamples=val_dataset, testExamples=test_dataset, numIters=5, eta=0.01, vocab_list=vocab_token_data, avg_correct_answer_length_in_train=average_correct_answer_length)
    

    print('Use weights of words to evaluate the ')


if __name__ == "__main__":
     tf.app.run()






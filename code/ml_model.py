# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random
import time
import logging

import numpy as np
from six.moves import xrange
from os.path import join as pjoin
import tensorflow as tf

from evaluate import exact_match_score, f1_score, evaluate
import sys
import math
import re

logging.basicConfig(level=logging.INFO)

def load_token_file(file_name):
    data = []
    file_contents = open(file_name, "rb")
    for line in file_contents:
        # Assumes already in the space delimited token index format
        data.append(line.rstrip()) # Get rid of trailing newline
    return data

def load_datasets():
    # Do what you need to load datasets from FLAGS.data_dir
    # We load the .ids. file because in qa_answer they are also loaded
    dataset_dir = FLAGS.data_dir
    abs_dataset_dir = os.path.abspath(dataset_dir)
    # We will no longer use the train dataset, we are only going to be using the val dataset and splitting it up into our own test, train, and val datasets
    # NOTE: get all the files that we want to load
    train_answer_file = os.path.join(abs_dataset_dir, "new_train_answer_data")
    train_context_file = os.path.join(abs_dataset_dir, "new_train_context_data")
    train_question_file = os.path.join(abs_dataset_dir, "new_train_question_data")
    test_answer_file = os.path.join(abs_dataset_dir, "new_test_answer_data")
    test_context_file = os.path.join(abs_dataset_dir, "new_test_context_data")
    test_question_file = os.path.join(abs_dataset_dir, "new_test_question_data")
    val_answer_file = os.path.join(abs_dataset_dir, "new_val_answer_data")
    val_context_file = os.path.join(abs_dataset_dir, "new_val_context_data")
    val_question_file = os.path.join(abs_dataset_dir, "new_val_question_data")
    vocab_file = os.path.join(abs_dataset_dir, "vocab.dat")
    
    demo_answer_file = os.path.join(abs_dataset_dir, "new_demo_answer_data")
    demo_context_file = os.path.join(abs_dataset_dir, "new_demo_context_data")
    demo_question_file = os.path.join(abs_dataset_dir, "new_demo_question_data")

    train_answer_data = load_token_file(train_answer_file)
    train_context_data = load_token_file(train_context_file)
    train_question_data = load_token_file(train_question_file)
    
    test_answer_data = load_token_file(test_answer_file)
    test_context_data = load_token_file(test_context_file)
    test_question_data = load_token_file(test_question_file)
    
    valid_answer_data = load_token_file(val_answer_file)
    valid_context_data = load_token_file(val_context_file)
    valid_question_data = load_token_file(val_question_file)
    
    vocab_token_data = load_token_file(vocab_file)
    
    demo_answer_data = load_token_file(demo_answer_file)
    demo_context_data = load_token_file(demo_context_file)
    demo_question_data = load_token_file(demo_question_file)
    
    # Merge data
    val_dataset = (valid_context_data, valid_question_data, valid_answer_data)
    test_dataset = (test_context_data, test_question_data, test_answer_data)
    train_dataset = (train_context_data, train_question_data, train_answer_data)
    
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

#Baseline: To get the context with question and answer
def baseline(train_dataset, vocab_token_data):
    train_context = train_dataset[0]
    train_question = train_dataset[1]
    train_answer = train_dataset[2]
    
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
    
        # VERSION 2: (GLoVE) Pick the phrase with the greatest cosine similarity in glove vectors for phrase and question
        #        for phrase in phrases:
        #            # gloveCosineSimilarityValue will be between [-1, 1]
        #            gloveCosineSimilarityValue = getGloveCosineSimilarityValue(question, phrase, glove_vector_data, vocab_token_data)
        #            if gloveCosineSimilarityValue > max_count:
        #                max_count = gloveCosineSimilarityValue
        #                max_phrase = phrase
        
        # Max phrase is now the predicted answer in baseline
        predicted_answer = max_phrase
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
        total_sum_of_word_ids_in_all_phrases += len(phrase.split())
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

def main(_):
    print ('Analyze the dataset: ')
    val_dataset, test_dataset, train_dataset, vocab_token_data, demo_dataset = load_datasets()
    
    val_context_data, val_question_data, val_answer_data = val_dataset
    #avg_num_of_word_ids_in_paragraphs = printAvgParagraphLength(valid_context_data)
    val_avg_num_of_words_in_paragraphs = printAvgLength(val_context_data)
    val_avg_num_of_words_in_questions = printAvgLength(val_question_data)
    val_avg_num_of_words_in_answers = printAvgLength(val_answer_data)
    #print('val_avg_num_of_words_in_paragraphs = ', val_avg_num_of_words_in_paragraphs, '\n',     'val_avg_num_of_words_in_questions = ', val_avg_num_of_words_in_questions, '\n', 'val_avg_num_of_words_in_answers = ', val_avg_num_of_words_in_answers, '\n')
    
    test_context_data, test_question_data, test_answer_data = test_dataset
    test_avg_num_of_words_in_paragraphs = printAvgLength(test_context_data)
    test_avg_num_of_words_in_questions = printAvgLength(test_question_data)
    test_avg_num_of_words_in_answers = printAvgLength(test_answer_data)
    #print('test_avg_num_of_words_in_paragraphs = ', test_avg_num_of_words_in_paragraphs, '\n',     'test_avg_num_of_words_in_questions = ', test_avg_num_of_words_in_questions, '\n', 'test_avg_num_of_words_in_answers = ', test_avg_num_of_words_in_answers, '\n')
    
    train_context_data, train_question_data, train_answer_data = train_dataset
    train_avg_num_of_words_in_paragraphs = printAvgLength(train_context_data)
    train_avg_num_of_words_in_questions = printAvgLength(train_question_data)
    train_avg_num_of_words_in_answers = printAvgLength(train_answer_data)
    #print('train_avg_num_of_words_in_paragraphs = ', train_avg_num_of_words_in_paragraphs, '\n',     'train_avg_num_of_words_in_questions = ', train_avg_num_of_words_in_questions, '\n', 'train_avg_num_of_words_in_answers = ', train_avg_num_of_words_in_answers, '\n')
    
    #print('\n', 'Baseline model and evaluation function on train dataset: ')
    #baseline(train_dataset, vocab_token_data)
    
    #print('\n', 'Baseline model and evaluation function on val dataset: ')
    #baseline(val_dataset, vocab_token_data)
    
    
    baseline(demo_dataset, vocab_token_data)


if __name__ == "__main__":
    tf.app.run()






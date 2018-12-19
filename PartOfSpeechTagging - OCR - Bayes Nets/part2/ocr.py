#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2018)
#

'''
This code is used to recognize the sentence in the given image.
Assumptions: the images have same size and font as training data

Current Approach:
1. Training data:
    it contains an image with all expected characters. we separate this image into seperate characters and store these images as np array in dictionary
2. calculating emission probability
    we calculate emission probability of each letter by calculating pixel by pixel difference of each character in test image with all characters in training data
    we then convert this error into probability and store probabilities for each character in training data corresponding to each character in trst data.
    we select characters with max probability for simplified output.
    we also use noise reduction to eliminate errors in borders of the image which is always white
3. Transmission probability
    we have considered novel 'Moby Dick; or The Whale' and training data from part1 as language sample to calculate transission probability.
    we filter the document to remove all characters other than expected characters.
    we consider transmission from one letter to another
4. Viterbi Algorithm.
    we use emission emission probability and transmission probability calculated in step 2 and 3 to find most probable sequence using vertibi algorithm.

Approaches tried to calculate probabilities

1. mean square error: we blur training image and test image then we calculate mean square error for each character. we convert this error into probability
2. Edge detection: we try to remove noise from the test image by performing edge detection on the test image sing SOBEL edge detection kernal to find edges and then using
    inverted output from sobel filter to perform OCR.


'''

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sys
import math
import numpy as np
from scipy.signal import convolve2d

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
blur = np.ones((3, 3))
blur /= np.sum(blur)
emission_prob_tab = dict()
MIN_PROB = 10e-9
TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
train_prob_tab = [MIN_PROB]*len(TRAIN_LETTERS)
transition_prob_tab = [[MIN_PROB]*len(TRAIN_LETTERS) for _ in range(len(TRAIN_LETTERS))]

'''
this function takes the image(training/testing) and separates the image in individual characters
these individual characters are blurred to remove noise from image using box blur kernel which is convolved with the character.
these characters are returned as list of arrays
'''


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    temp = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        temp += [[[1 if px[x, y] < 1 else 0 for x in range(x_beg, x_beg + CHARACTER_WIDTH)]
                  for y in range(0, CHARACTER_HEIGHT)], ]
    return np.array(temp)


def load_training_letters(fname):
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


def calculate_emission_prob(train_letters, test_letters):
    '''
    this function takes all characters of test data and calculates emission probablity with all characters of training data
    emission probablity is calculated by taking mean squared error between a character and all training characters.
    0 means exact match. lower values of error represent similar images.
    we convert this to probabilities and return dictionary with all characters of testing data as keys and list of all characters
    from training data with its probabilities as values
    '''

    # test_image, true_error
    bb, bw, wb, ww = 0, 0.9, 0.1, 0

    result = dict()
    num = 0
    temp = []
    ans = ''
    for letter_mat in test_letters:
        temp_t = []
        min_error = np.inf
        for t_letter, t_letter_mat in train_letters.items():
            err = np.sum((letter_mat == 1) * (t_letter_mat == 1)) * bb + np.sum((letter_mat == 1) * (t_letter_mat == 0)) * \
                bw + np.sum((letter_mat == 0) * (t_letter_mat == 1)) * wb + \
                np.sum((letter_mat == 0) * (t_letter_mat == 0)) * ww
            prob = 1/(err + 1)
            temp_t += [[t_letter, prob]]
            if err < min_error:
                min_error = err
                best_letter = t_letter
                ans += best_letter
        sum_err = sum(t[1] for t in temp_t)
        for i in range(0, len(temp_t)):
            temp_t[i][1] = temp_t[i][1]/sum_err
        temp += [max(temp_t, key=lambda x: x[1])]
        result[num] = temp_t
        num += 1
    sentence = "".join(x[0] for x in temp)
    print("Simple:  ", sentence)
    return result


def calculate_transition_prob(train_txt_fname):
    # TODO: Improve reading of data
    book = open(train_txt_fname, "r", encoding="UTF-8")
    data = ""
    for c in book.read():
        if c in TRAIN_LETTERS:
            # if wasPreviousStop and c != " ":
            #     # increment beginning probablity for class
            #     wasPreviousStop = False
            # elif c == "." or c =="\n":
            #     wasPreviousStop = True
            data += c

    for i in range(0, len(data) - 1):
        # TODO: Is this right? Because transition probablity is calculated as
        # prev to current from the observed. But here we are adding it to the train to train
        # but we don't know which letter is it in the test.
        curr = TRAIN_LETTERS.index(data[i])
        next = TRAIN_LETTERS.index(data[i+1])
        transition_prob_tab[curr][next] += 1

        # Also update the letter probablity
        train_prob_tab[curr] += 0.1

    for row in transition_prob_tab:
        t_sum = sum(row)
        for i in range(0, len(row)):
            row[i] /= t_sum

    sum_train = sum(train_prob_tab)
    for i in range(len(train_prob_tab)):
        train_prob_tab[i] /= sum_train

    # Reducing probability for space
    train_prob_tab[TRAIN_LETTERS.index(" ")] *= 0.2


def hmm_viterbi(len_test_letters):
    """
    Calulates maximum a posteriori (MAP) labeling for the given test_letters using
    Viterbi algorithm.
    We calculate probability of any word(currW) being associated with any tag
    (c_train_letter) as:
    v[c_train_letter][test_letter] = -math.log(emission_prob_tab[test_letter][c_train_letter][1])
    """

    # for all the words which were not found in training data, we assign a low
    # emission probability
    MISSING_PROB = -math.log(10e-9)
    len_TRAIN_LETTERS = len(TRAIN_LETTERS)
    v = np.full((len_TRAIN_LETTERS, len_test_letters), MISSING_PROB)
    vPrev = np.full((len_TRAIN_LETTERS, len_test_letters), 0)

    # initialize for first letter
    # emission_prob_tab is a dictionary which holds emission probablity for all
    # characters in the test string, in the following format:
    # index(in the test string) : [[A,p(A)],[B,p(B)]....]
    c_train_letter, test_letter = (0, 0)
    for c_train_letter in range(0, len_TRAIN_LETTERS):
        v[c_train_letter][test_letter] = - \
            (math.log(emission_prob_tab[test_letter][c_train_letter][1]))*1000

    for test_letter in range(1, len_test_letters):
        for c_train_letter in range(0, len_TRAIN_LETTERS):
            # For each of the current training character, find the max transition
            # from all other characters.
            minProduct = np.inf
            minT = -1
            for p_train_letter in range(0, len_TRAIN_LETTERS):
                currProduct = v[p_train_letter][test_letter-1] - \
                    (math.log((transition_prob_tab[p_train_letter][c_train_letter])))
                if currProduct < minProduct:
                    minProduct = currProduct
                    minT = p_train_letter

            v[c_train_letter][test_letter] = minProduct - \
                (math.log(emission_prob_tab[test_letter][c_train_letter][1]))*1000

            # update vPrev to hold previous tag which gives best probability
            vPrev[c_train_letter][test_letter] = minT

    v_sentence = ""
    c_train_letter = v.argmin(axis=0)[-1]
    v_sentence = TRAIN_LETTERS[c_train_letter] + v_sentence
    for test_letter in range(len_test_letters-1, 0, -1):
        # search for the previous letter which gave you the max value
        # vPrev stores index of previous letter that we have to pick
        c_train_letter = vPrev[c_train_letter][test_letter]
        v_sentence = TRAIN_LETTERS[c_train_letter] + v_sentence
    return v_sentence


def remove_noise(letters, prob):
    result = []
    for letter in letters:
        result += [letter * prob]

    result = np.array(result)
    return result


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]

train_image = load_letters(train_img_fname)
pixel_prob = train_image.sum(axis=0)
pixel_prob[pixel_prob > 3] = 1


train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

# test_letters = remove_noise(test_letters, pixel_prob)

calculate_transition_prob(train_txt_fname)
emission_prob_tab = calculate_emission_prob(train_letters, test_letters)
viterbi_result = hmm_viterbi(len(test_letters))
print("viterbi: ", viterbi_result)
print("Final Answer:")
print(viterbi_result)

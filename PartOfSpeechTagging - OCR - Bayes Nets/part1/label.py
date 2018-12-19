#!/usr/bin/env python3
###################################
# CS B551 Fall 2018, Assignment #3
# D. Crandall
#
# There should be no need to modify this file, although you
# can if you really want. Edit pos_solver.py instead!
#
# To get started, try running:
#
#   python ./label.py bc.train bc.test.tiny
#

from pos_scorer import Score
from pos_solver import *
import sys

# Read in training or test data file
#


def read_data(fname):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [(data[0::2], data[1::2]), ]
    return exemplars


####################
# Main program
#

if len(sys.argv) < 3:
    print("Usage: \n./label.py training_file test_file")
    sys.exit()

(train_file, test_file) = sys.argv[1:3]

print("Learning model...")
solver = Solver()
train_data = read_data(train_file)
solver.train(train_data)

print("Loading test data...")
test_data = read_data(test_file)

print("Testing classifiers...")
scorer = Score()

Algorithms = ("Simple", "HMM", "Complex")
Algorithm_labels = [str(i+1) + ". " + Algorithms[i] for i in range(0, len(Algorithms))]
for (s, gt) in test_data:

    outputs = {"0. Ground truth": gt}

    # run all algorithms on the sentence
    for (algo, label) in zip(Algorithms, Algorithm_labels):
        outputs[label] = solver.solve(algo, s)

    # calculate posteriors for each output under each model
    posteriors = {o: {a: solver.posterior(a, s, outputs[o]) for a in Algorithms} for o in outputs}
    Score.print_results(s, outputs, posteriors, Algorithms)

    scorer.score(outputs, gt)
    scorer.print_scores()

    print("----")

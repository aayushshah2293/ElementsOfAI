###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
# Bhushan Malgaonkar: bmalgaon
# Mahesh Belnekar: mbelnek
# Aayush Shah: aaymshah
#
# (Based on skeleton code by D. Crandall)
#
#
####
#
# We train the data and first calculate emission and transition probablity tables.
# Using these values we calculate simplified, complex and hmm models.
#
# NOTE: Posterior probablities are -Log() of the actual probablities so, probablities
# are positive. Lower the value, better the result.

# Simplified
# The tag entirely depends on the word itself, so we can choose tag for a given word such that P(tag|word) is maximum
# P(tag|word) = P(word|tag) * P(tag), these two probabilities can be calculated from the data
#
# HMM Viterbi
# Calulates maximum a posteriori (MAP) labeling for the given sentence using
# Viterbi algorithm.
# We calculate probability of any word(currW) being associated with any tag
# (currT) as:
# v[currT][w] = v[prevT][w-1]*transitionTab[prevT][currT]*emissionTab[currW][currT]
#
# Complex MCMC
# Calculate likelyhood of tags (S1, S2, S3, S4...) for sentence (W1, W2, W3, W4...) using bayes theorem
# likelyhood = P(S1) * P(S2|S1) * P(S3|S2,S1) * P(S4|S3,S2) * ...
#                * P(S1|W1) * P(S2|W2) * P(S3|W3) * P(S4|W4) * ...
# more detailed explanation in comments
####

import random
import math
import numpy as np
from collections import deque


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        """
        - tagLookUp provides the index of the current tag(given by the key)
        in tables emissionTab abd transitionTab
        - emissionTab stores the emission probability
        - transitionTab stores the transaction probability
        """
        self.tagLookUp = {"adj": 0, "adv": 1, "adp": 2, "conj": 3, "det": 4, "noun": 5,
                          "num": 6, "pron": 7, "prt": 8, "verb": 9, "x": 10, ".": 11}
        self.tagIndexLookUp = {0: "adj", 1: "adv", 2: "adp", 3: "conj", 4: "det", 5: "noun",
                               6: "num", 7: "pron", 8: "prt", 9: "verb", 10: "x", 11: "."}
        self.num_tags = len(self.tagLookUp)
        self.emissionTab = dict()
        self.mincount = 1/100000
        self.transitionTab = [[self.mincount]*len(self.tagLookUp)
                              for _ in range(len(self.tagLookUp))]
        self.transition2Tab = np.zeros(
            (len(self.tagLookUp), len(self.tagLookUp), len(self.tagLookUp)))
        # probability of all tags
        self.tagTab = [0 for _ in range(len(self.tagLookUp))]
        self.word_prob = {}
        self.word_transition = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling.
    # NOTE: Posterior probablities are -Log() of the actual probablities so, probablities
    # are positive. Lower the value, better the result.
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.evaluate_simplified([self.tagLookUp[x] for x in label], sentence)
        elif model == "Complex":
            return self.evaluate_mcmc([self.tagLookUp[x] for x in label], sentence, denominator=False)
        elif model == "HMM":
            return self.evaluate_hmm([self.tagLookUp[x] for x in label], sentence)
        else:
            print("Unknown algo!")

    def calculate_word_count(self, data):
        for words, tags in data:
            for word in words:
                if word not in self.word_prob:
                    self.word_prob[word] = 0
                self.word_prob[word] += 1
        total = sum(self.word_prob.values())
        self.word_prob = {k: v/total for k, v in self.word_prob.items()}

    def get_word_prob(self, word):
        MISSING_PROB = 10e-9
        return self.word_prob[word] if word in self.word_prob else MISSING_PROB

    def calculate_word_transition(self, data):
        for words, tags in data:
            for i in range(len(words) - 1):
                if words[i] not in self.word_transition:
                    self.word_transition[words[i]] = {}
                if words[i+1] not in self.word_transition[words[i]]:
                    self.word_transition[words[i]][words[i+1]] = 0
                self.word_transition[words[i]][words[i+1]] += 1

        for from_word in self.word_transition:
            total = sum(self.word_transition[from_word].values())
            self.word_transition[from_word] = {k: v/total for k,
                                               v in self.word_transition[from_word].items()}

    def get_word_transition(self, from_word, to_word):
        MISSING_PROB = 10e-12
        return self.word_transition[from_word][to_word] if from_word in self.word_transition and to_word in self.word_transition[from_word] else MISSING_PROB

    # Update transitionTab by calculating transaction probability of all the tags
    def calculate_transition_prob(self, data):
        for tup in data:
            for i in range(0, len(tup[1])-1):
                # print(tup[1][i])
                curr = self.tagLookUp[tup[1][i]]
                next = self.tagLookUp[tup[1][i+1]]
                self.transitionTab[curr][next] += 1

        for row in self.transitionTab:
            t_sum = sum(row)
            for i in range(len(row)):
                row[i] = row[i]/t_sum

        return True

    # Update transitionTab by calculating transaction probability of all the tags
    def calculate_transition2_prob(self, data):
        MISSING_PROB = 10e-9

        for words, tags in data:
            for i in range(2, len(words)):
                tag_i = self.tagLookUp[tags[i]]
                tag_i_1 = self.tagLookUp[tags[i-1]]
                tag_i_2 = self.tagLookUp[tags[i-2]]
                self.transition2Tab[tag_i][tag_i_1][tag_i_2] += 1

        # Convert into probabilities
        sum = np.sum(self.transition2Tab, axis=0)

        # cases where P(S3|S2,S1) is 0 for all combinations of S2,S1.
        # we can handle this case by assigning sum = infinity so that corresponding probability becomes zero
        sum[sum == 0] = np.inf

        # divide by total sum to get probabilities
        self.transition2Tab /= sum

        # Replace zero probabilities with MISSING_PROB
        self.transition2Tab[self.transition2Tab == 0] = MISSING_PROB

    # calculates emmission probability from from training data and stores in emmissiontab dictionary
    def calculate_emission_prob(self, data):
        for (s, t) in data:
            for i in range(0, len(s)):
                if s[i] in self.emissionTab:
                    temp = self.emissionTab[s[i]]
                    temp[self.tagLookUp[t[i]]] += 1
                    self.emissionTab[s[i]] = temp

                else:
                    temp = [self.mincount]*12
                    temp[self.tagLookUp[t[i]]] += 1
                    self.emissionTab[s[i]] = temp

        for word in self.emissionTab:
            for tag in self.tagLookUp.values():
                self.tagTab[tag] += self.emissionTab[word][tag]
        sum_tags = sum(self.tagTab)
        self.tagTab = [self.tagTab[i]/sum_tags for i in range(0, len(self.tagTab))]

        for word in self.emissionTab:
            count = self.emissionTab[word]
            word_sum = sum(count)
            for j in range(0, len(count)):
                count[j] = count[j]/word_sum
            self.emissionTab[word] = count

    def get_emission_prob(self, word, tag_index):
        MISSING_PROB = 10e-9
        return self.emissionTab[word][tag_index] if word in self.emissionTab else MISSING_PROB

    # Do the training!
    # Process training data to calculate transition and emission probability
    # of tags and store them in tables transitionTab and emissionTab respectively.
    def train(self, data):
        result = self.calculate_transition_prob(data)
        self.calculate_emission_prob(data)
        self.calculate_transition2_prob(data)
        self.calculate_word_count(data)

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #

    def evaluate_simplified(self, tags, sentence):
        posterior = 0
        for i in range(len(sentence)):
            posterior -= math.log(self.get_emission_prob(sentence[i], tags[i]))
            posterior -= math.log(self.tagTab[tags[i]])
            # posterior += math.log(self.get_word_prob(sentence[i]))
        return posterior

    def simplified(self, sentence):
        """
        Calculates P(tag|word) = P(word|tag) * P(tag) for each tag, for each word in the sentence
        Returns a list of most likely tags for all the words independent of their position in the given sentence
        """

        tags = []
        for word in sentence:
            min_cost = float('inf')
            best_tag = None
            for tag_name, tag_index in self.tagLookUp.items():
                cost = -math.log(self.get_emission_prob(word, tag_index)) - \
                    math.log(self.tagTab[tag_index])
                if cost < min_cost:
                    min_cost = cost
                    best_tag = tag_name
            tags.append(best_tag)
        return tags

    # Start of Complex MCMC

    def evaluate_mcmc(self, tags, sentence, denominator=False):
        """
        Calculates likelyhood of tags (S1, S2, S3, S4...) for sentence (W1, W2, W3, W4...) using bayes theorem
        likelyhood = P(S1) * P(S2|S1) * P(S3|S2,S1) * P(S4|S3,S2) * ...
                        * P(S1|W1) * P(S2|W2) * P(S3|W3) * P(S4|W4) * ...

        To keep calculations within bounds use -log of all probabilities

        """
        likelyhood = 0

        if len(sentence) > 0:
            likelyhood += -math.log(self.tagTab[tags[0]])
        if len(sentence) > 1:
            likelyhood += -math.log(self.transitionTab[tags[0]][tags[1]])
        for i in range(2, len(sentence)):
            likelyhood += -math.log(self.transition2Tab[tags[i], tags[i-1], tags[i-2]])

        for i in range(len(sentence)):
            likelyhood += -math.log(self.get_emission_prob(sentence[i], tags[i]))

        if denominator:
            if len(sentence) > 0:
                likelyhood -= -math.log(self.get_word_prob(sentence[0]))
            for i in range(1, len(sentence)):
                likelyhood -= -math.log(self.get_word_transition(sentence[i - 1], sentence[i]))

        return likelyhood

    def prob_dist_for_word_i(self, tags, sentence, i):
        """
        Calculates probability distribution over each tag assigned to word at index i, keeping all other tags same.
        Returns an array of probabilities for each tag as per indexes specified in tagLookUp
        """
        likelyhood = np.zeros(len(self.tagLookUp))
        backup = tags[i]

        for tag_index in range(len(self.tagLookUp)):
            tags[i] = tag_index

            # store the value at index corresponding to that tag
            likelyhood[tag_index] = self.evaluate_mcmc(tags, sentence)

        tags[i] = backup

        # Convert likelyhood into probability distribution to make sampling easier

        # subtracting minimum is same as dividing probabilities, which keeps the relative probabilities the same
        # 10 times more likely event will still be 10 times more likely
        likelyhood -= np.min(likelyhood)

        # convert logs into probabilities, since now they are within reasonable bounds
        prob_dist = np.exp(-likelyhood)
        prob_dist /= sum(prob_dist)

        return prob_dist

    def complex_mcmc(self, sentence):
        NUM_ITER_GIBBS = 100
        # gibbs sampling

        # initialize with first tag
        # tags = np.zeros(len(sentence), dtype='int')

        # initialize using simplified --> yields similar accuracy
        tags = np.array([self.tagLookUp[x] for x in self.simplified(sentence)])

        for iter in range(NUM_ITER_GIBBS):
            # Choose a word at random from the sentence
            word_i = np.random.randint(0, len(sentence))

            # Choose word cyclically --> yields similar accuracy
            # word_i = iter % len(sentence)

            prob_dist = self.prob_dist_for_word_i(tags, sentence, word_i)

            # Select randomly over given new probability distribution
            tag_i = np.random.choice(np.arange(len(prob_dist)), p=prob_dist)
            tags[word_i] = tag_i

        # sampling may randomly (with very low probability) pick very unlikely tag for a particular word, therefore
        # return most likely tag for each word using generated posterior probabilities
        for itr in range(5):
            for i in range(len(tags)):
                tags[i] = np.argmax(self.prob_dist_for_word_i(tags, sentence, i))

        return [self.tagIndexLookUp[x] for x in tags]

    # End of Complex MCMC

    def evaluate_hmm(self, tag, sentence):
        '''
        Calculate posterior for HMM as:
        P(tag|word) = Emission(word|tag)*Transition(prev_tag|curr_tag)
        '''

        # For the fist word, we do not consider
        post = - math.log(self.get_emission_prob(sentence[0], tag[0]))

        for i in range(len(sentence)):
            post -= math.log(self.get_emission_prob(sentence[i], tag[i]))
            post -= math.log(self.transitionTab[tag[i-1]][tag[i]])

        return post

    def hmm_viterbi(self, sentence):
        """
        Calulates maximum a posteriori (MAP) labeling for the given sentence using
        Viterbi algorithm.
        We calculate probability of any word(currW) being associated with any tag
        (currT) as:
        v[currT][w] = v[prevT][w-1]*transitionTab[prevT][currT]*emissionTab[currW][currT]
        """
        # TODO:
        # 1. Sentences probability is still low(<50%)
        # 2. Implement posterior probability

        # for all the words which were not found in training data, we assign a low
        # emission probability
        MISSING_PROB = -math.log(10e-9)
        v = np.full((len(self.tagLookUp), len(sentence)), MISSING_PROB)
        vPrev = np.full((len(self.tagLookUp), len(sentence)), 0)

        # initialize for first word
        w, currW = (0, sentence[0])
        if currW in self.emissionTab:
            for currT in range(0, len(self.tagLookUp)):
                v[currT][w] = -math.log(self.emissionTab[currW][currT])

        for w in range(1, len(sentence)):
            currW = sentence[w]
            for currT in range(len(self.tagLookUp)):
                # For each of the current tag, find the max checking transition
                # from all other tags.
                minProduct = np.inf
                minT = 0
                for prevT in range(len(self.tagLookUp)):
                    currProduct = v[prevT][w-1] - math.log(self.transitionTab[prevT][currT])
                    if currProduct < minProduct:
                        minProduct = currProduct
                        minT = prevT

                if currW in self.emissionTab:
                    v[currT][w] = minProduct - math.log(self.emissionTab[currW][currT])
                else:
                    v[currT][w] = minProduct - MISSING_PROB
                # update vPrev to hold previous tag which gives best probability
                vPrev[currT][w] = minT

        tags = deque([])
        currT = v.argmin(axis=0)[-1]
        tags.appendleft(self.tagIndexLookUp[currT])
        for w in range(len(sentence)-1, 0, -1):
            # search for the previous tag which gave you the max value
            # vPrev stores tag index of previous tag that we have to pick
            currT = vPrev[currT][w]
            tags.appendleft(self.tagIndexLookUp[currT])

        return list(tags)

    def preprocess(self, sentence):
        return [word[:-2] if word.endswith('\'s') else word for word in sentence]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        # sentence = self.preprocess(sentence)

        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

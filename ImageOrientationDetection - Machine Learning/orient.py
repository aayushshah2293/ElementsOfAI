#!/usr/bin/env python3
import sys
import numpy as np
import pickle
import random
import itertools
import math
from copy import deepcopy
from heapq import heappush, heapreplace
from collections import Counter

ORIENTATIONS = [0, 90, 180, 270]


class nearest():
    def __init__(self, K=40):
        # K=40 is found by testing various different Ks
        self.K = K
        self.train_image_array = None
        self.train_image_orientation = None

    """
    Maintain a max-heap of size k.
    While inserting new element, if heap size is less than k, insert without any checks.
    Otherwise, compare the new element with top (=max) element in the heap. if new element is larger replace top with it, else ignore the new element.

    Python only offers min-heap implementation. We can insert negative of actual element for it to act as max-heap.

    In addition to array element, we also need to insert the corresponding indexes.
    We can do this by making a tuple (element, index) since heap algorithms compare first element of tuple before 2nd.
    """

    def k_smallest(self, arr):
        max_heap = []
        for i, a in enumerate(arr):
            if len(max_heap) < self.K:
                heappush(max_heap, (-a, i))
            elif -a > max_heap[0][0]:
                heapreplace(max_heap, (-a, i))
        return [x[1] for x in max_heap]

    """
    Subtract test image array from all training images to get absolute error with each image.
    Using this difference, find k training images nearest to test image.
    Return the most frequent orientation for these k images
    """

    def find_orientation(self, test_image):
        diff = np.sum(np.abs(self.train_image_array - test_image)**2, axis=1)
        ksm = self.k_smallest(diff)
        return Counter(self.train_image_orientation[ksm]).most_common(1)[0][0]

    def load(self, trained_model):
        self.train_image_array, self.train_image_orientation = trained_model

    """
    Training for knn is simply saving the training data. Since during test we compare any test image with each training image.
    """

    def train(self, image_array, image_orientation):
        return (image_array, image_orientation)

    """
    Finds orientation for each image in the testing set and calculates accuracy by comparing it with actual orientation
    """

    def test(self, test_image_name, test_image_array, test_image_orientation):
        predicted = [self.find_orientation(x) for x in test_image_array]
        accuracy = np.sum(predicted == test_image_orientation) / len(test_image_orientation) * 100
        return test_image_name, predicted, accuracy

'''
citation:
The approach for adaboost is adapted from following link:
https://github.com/pulkitmaloo/Image-Orientation-Classification
However we have created the our own implementation using similar approach.
'''
class adaboost():
    sample_set_size = 500
    possible_orientation = 4
    possible_pairs_count = 6
    no_of_pixels = 192

    def __init__(self):
        pass

    def train(self, image_pairs, image_orientation_pairs, sample_set):
        curr_orientations = list(set(image_orientation_pairs))
        classifier = dict()
        # initialising weights to eaqual weights for wach image
        weights = np.array([(float(1)/float(len(image_orientation_pairs)))]
                           * len(image_orientation_pairs))
        for sample in sample_set:
            err = 0.0
            px1 = sample[0]
            px2 = sample[1]
            classified = []
            positive_classified_count = dict([(curr_orientations[i], 0)
                                              for i in range(0, len(curr_orientations))])
            negative_classified_count = dict([(curr_orientations[i], 0)
                                              for i in range(0, len(curr_orientations))])
            for k in range(0, len(image_pairs)):
                # classifing each image as positive or negative depending upon randomly selected sample
                if image_pairs[k][px1] >= image_pairs[k][px2]:
                    classified += ['positive']
                    positive_classified_count[image_orientation_pairs[k]] += 1
                else:
                    classified += ['negative']
                    negative_classified_count[image_orientation_pairs[k]] += 1
            # setting positive,negative orientation with orientation with highest number of images classified correctly
            pos_orientation = max(positive_classified_count,
                                  key=lambda k: positive_classified_count[k])
            neg_orientation = max(negative_classified_count,
                                  key=lambda k: negative_classified_count[k])
            # updating classified results to its corresponding positive or negative orientation
            classified_orientation = []
            classified_orientation += [pos_orientation if classified[i] ==
                                       'positive' else neg_orientation for i in range(0, len(image_orientation_pairs))]

            # calculating error
            for j in range(0, len(image_orientation_pairs)):
                if classified_orientation[j] != image_orientation_pairs[j]:
                    err += weights[j]
            # if error is greater than 0.5 the stumb is neglected since it introduces more error in the classifier providing less accuracy
            if err > 0.5:
                continue

            # reducing weights of correctly classified images
            for i in range(0, len(image_orientation_pairs)):
                if classified_orientation[i] == image_orientation_pairs[i]:
                    weights[i] = weights[i]*err/(1-err)
            # normalising weights
            weights_sum = sum(weights)
            weights = weights/weights_sum
            classifier[sample] = {}
            classifier[sample]['weight'] = math.log((1-err)/err)
            classifier[sample]['pos_orientation'] = pos_orientation
            classifier[sample]['neg_orientation'] = neg_orientation
        return [classifier, curr_orientations]

    def test(self, image_array, image_orientation, classifiers, curr_orientations):
        orientation_sign = dict()
        orientation_sign[curr_orientations[0]] = 1
        orientation_sign[curr_orientations[1]] = -1
        image_classification = [0]*len(image_orientation)
        for classifier in classifiers:
            px1 = classifier[0]
            px2 = classifier[1]
            classifier_dict = classifiers[classifier]
            for i in range(len(image_orientation)):
                if image_array[i][px1] >= image_array[i][px2]:
                    orientation = classifier_dict['pos_orientation']
                    image_classification[i] += orientation_sign[orientation] * \
                        classifier_dict['weight']
                else:
                    orientation = classifier_dict['neg_orientation']
                    image_classification[i] += orientation_sign[orientation] * \
                        classifier_dict['weight']
        classification = []
        for i in range(len(image_orientation)):
            if image_classification[i] >= 0:
                classification += [curr_orientations[0]]
            else:
                classification += [curr_orientations[1]]
        return classification


class Forest:
    class Node:
        '''
        Node class use to store the structure of decision tree.
        Class Members:
         - predicate - [Pixel position,val]
         - orientation - None by default
         - left - left child
         - right - right child
        '''

        def __init__(self, predicate, orientation=None, left=None, right=None):
            self.predicate = predicate
            self.orientation = orientation
            self.left = left
            self.right = right
    # end of node class

    def __init__(self, image_name, image_array, image_orientation):
        # Update these when you get train data
        self.total_set_size = 40000
        self.no_of_pixels = 192
        self.image_name = image_name
        self.image_array = image_array
        self.image_orientation = image_orientation

    # print the forest
    def printTree(self, node):
        if not node:
            return
        print("predicate: ", node.predicate, " orientation: ", node.orientation)
        self.printTree(node.left)
        self.printTree(node.right)

    # returns the orientation of the given images based of the decision tree root
    @staticmethod
    def get_orientation_using_tree(root, image):
        if not root:
            return -1

        # check if it is a leaf node
        if not root.predicate:
            return root.orientation

        (pos, val) = root.predicate
        if image[pos] <= val:
            # go left
            temp_o = Forest.get_orientation_using_tree(root.left, image)
        else:
            # go right
            temp_o = Forest.get_orientation_using_tree(root.right, image)

        # if you get orientation as -1 from the child node, it means that the
        # child node does not exist. In that case, you send your own orientation
        if temp_o == -1 or temp_o == None:
            temp_o = root.orientation

        return temp_o

    # returns a dictionary of predicates, where
    # key = pixel position and val = threshold
    def get_predicate_set(self, length):
        predicate_set = set()
        # TODO: Later we'll always add corner nodes, so we might need to update
        # the boundaries here.
        # We create a set of possible predicates, here we create 2 random tuples for each
        # position at 64,128,192
        for i in range(0, length):
            predicate_set.add(tuple([i, random.randint(64, 128)]))
            predicate_set.add(tuple([i, random.randint(120, 192)]))
        return predicate_set

    def get_best_orientation(self, dataset_index_list):
        # We keep a count of each orientation
        c_total_dict = {0: 0, 90: 0, 180: 0, 270: 0}
        for d in dataset_index_list:
            c_total_dict[self.image_orientation[d]] += 1

        # return the orientation which appears maximum no.of times
        return max(c_total_dict, key=c_total_dict.get)

    # construct a decision tree using current set of predicates and training subset
    def construct_decision_tree(self, dataset_index_list, predicate_set, entropy):
        len_data_set = len(dataset_index_list)
        overfit_threshold = 5
        best_orientation = self.get_best_orientation(dataset_index_list)

        # We define a threshold, if we find any set which has lower entropy
        # we consider its best orientation and stop dividing further
        entropy_threshold = 0.15

        # if the entropy of the current dataset is good enough, we end it here
        if entropy <= entropy_threshold or (not predicate_set):
            return self.Node(None, best_orientation)

        least_entropy, best_predicate, best_l, best_l_e, best_r, best_r_e = (
            float('inf'), None, None, None, None, None)
        not_good_predicates = []
        for (p, v) in predicate_set:
            left_tree, right_tree = self.split_data(dataset_index_list, p, v)

            # In any of the trees are empty, then we actually haven't achieved anything
            if len(left_tree) < overfit_threshold or len(right_tree) < overfit_threshold:
                # delete all these
                not_good_predicates.append((p, v))
                continue

            l_entropy, r_entropy = (self.calculate_entropy(left_tree),
                                    self.calculate_entropy(right_tree))

            # get average entropy, as our branching factor is only 2, we'll
            # just have to take average of 2 numbers
            avg_entropy = (len(left_tree)*l_entropy) + (len(right_tree)*r_entropy)
            avg_entropy /= len_data_set

            # compare with least entropy
            if avg_entropy < least_entropy:
                least_entropy = avg_entropy
                best_predicate = (p, v)
                best_l = left_tree
                best_r = right_tree
                best_l_e = l_entropy
                best_r_e = r_entropy

        # check if we have found any good predicate, if not, we return None
        if not best_predicate:
            return None

        # delete all not useful predicates
        for n in not_good_predicates:
            predicate_set.discard(n)

        # we delete the chosen predicate before we move deeper, but maintain original copy
        t_predicate_set = deepcopy(predicate_set)
        t_predicate_set.discard(best_predicate)

        l_child = self.construct_decision_tree(best_l, t_predicate_set, best_l_e)
        r_child = self.construct_decision_tree(best_r, t_predicate_set, best_r_e)

        # return a node with these as children
        return self.Node(best_predicate, best_orientation, l_child, r_child)

    # returns the entropy of the current dataset
    def calculate_entropy(self, dataset_index_list):
        entropy = 0
        n_total = len(dataset_index_list)

        # We keep a count of each orientation
        c_total_dict = {0: 0, 90: 0, 180: 0, 270: 0}
        for d in dataset_index_list:
            c_total_dict[self.image_orientation[d]] += 1

        for count in c_total_dict.values():
            if count > 0:
                f = count/n_total
                entropy -= f*math.log(f)
        return entropy

    # splits the data, given the predicate(i.e. pixel position and threshold)
    def split_data(self, dataset_index_list, p, v):
        left_tree, right_tree = list(), list()

        for d in dataset_index_list:
            if self.image_array[d][p] <= v:
                left_tree.append(d)
            else:
                right_tree.append(d)

        return left_tree, right_tree


def genrate_output(image_names, final_orientation):
    output = open('output.txt', 'w')
    for i in range(len(image_names)):
        output.write(image_names[i] + " " + str(final_orientation[i]) + "\n")
    output.close()


def images_dictionary_on_orientations(image_array, image_orientation):
    images = dict([(ORIENTATIONS[i], []) for i in range(len(ORIENTATIONS))])
    images_orientation = dict([(ORIENTATIONS[i], []) for i in range(len(ORIENTATIONS))])
    for i in range(0, len(image_orientation)):
        orientation = image_orientation[i]
        temp = image_array[i]
        images[orientation] += [image_array[i]]
        images_orientation[orientation] += [orientation]

    return images, images_orientation


def create_pairs(no_of_pixels):
    pairs = []
    for i in range(no_of_pixels):
        for j in range(i, no_of_pixels):
            if i != j:
                pairs += [(i, j)]
    return pairs


def read_file(file_name):
    temp = np.loadtxt(file_name, dtype=str)
    image_name = temp[:, 0]
    image_orientation = temp[:, 1].astype(int)
    image_array = temp[:, 2:].astype(int)
    return image_name, image_array, image_orientation


# Main function
if len(sys.argv) < 5 or len(sys.argv) >= 6:
    print("the argument should be in format [mode] [train/test filename] [model_file] [model]")
    sys.exit(-1)
action, file_name, model_file_name, model = sys.argv[1:]
image_name, image_array, image_orientation = read_file(file_name)

if action == 'train':
    print("Training Mode:", model, "\nTraining in progress...")
    trained_model = []
    if model == "nearest":
        # K-nearest algorith
        trained_model = nearest().train(image_array, image_orientation)
    elif model == "adaboost":
        # Adaboost Algorithm
        boost = adaboost()
        universal_sample_set = create_pairs(boost.no_of_pixels)
        sample_set = []
        sample_set += [random.sample(universal_sample_set, boost.sample_set_size)
                       for i in range(boost.possible_pairs_count)]
        images, images_orientation = images_dictionary_on_orientations(
            image_array, image_orientation)
        image_pairs = []
        images_orientation_pairs = []
        for O1, O2 in itertools.combinations(ORIENTATIONS, 2):
            image_pairs += [images[O1] + images[O2]]
            images_orientation_pairs += [images_orientation[O1] + images_orientation[O2]]

        pairs_classifier_forest = []
        for i in range(boost.possible_pairs_count):
            pairs_classifier_forest += [boost.train(image_pairs[i],
                                                    images_orientation_pairs[i], sample_set[i])]
        trained_model = (pairs_classifier_forest, boost)

    elif model == "forest" or model == "best":
        # random forest
        no_of_trees = 200

        tree_list = list()
        forest = Forest(image_name, image_array, image_orientation)
        for i in range(0, no_of_trees):
            n = forest.construct_decision_tree(random.sample(range(1, len(image_name)),
                                                             len(image_name)//30), forest.get_predicate_set(192), 999)
            if n:
                tree_list.append(n)
        trained_model = tree_list

    pickle.dump(trained_model, open(model_file_name, "wb"), protocol=3)
else:
    print("Testing Mode", model)
    accuracy = 0
    trained_model = pickle.load(open(model_file_name, "rb"))
    if model == "nearest":
        # K-nearest algorith
        # K-nearest algorith
        classifier = nearest()
        classifier.load(trained_model)
        image_name, predicted, accuracy = classifier.test(
            image_name, image_array, image_orientation)
        genrate_output(image_name, predicted)
    elif model == "adaboost":
        # Adaboost Algorithm
        pairs_classifier_forest, boost = trained_model
        classification = []
        for i in range(boost.possible_pairs_count):
            classifiers = pairs_classifier_forest[i][0]
            curr_orientations = pairs_classifier_forest[i][1]
            classification += [boost.test(image_array, image_orientation,
                                          classifiers, curr_orientations)]
        final = []
        correct_count = 0

        for i in range(len(image_orientation)):
            classified_orientations = []
            for j in range(boost.possible_pairs_count):
                classified_orientations += [classification[j][i]]
            final += [max(classified_orientations, key=classified_orientations.count)]
            if final[i] == image_orientation[i]:
                correct_count += 1
        accuracy = round((correct_count/len(image_orientation)), 2) * 100
        genrate_output(image_name, final)

    elif model == "forest" or model == "best":
        # random forest
        tree_list = trained_model
        total_set_size = len(image_name)
        total_correct = 0
        final_prediction_list = list()
        for i in range(total_set_size):
            true_orientation = image_orientation[i]
            pred_orientation = 0

            # we create a vote dictionary, to calculate vote of all tree
            # our model tends to favour 0 and 180, thats why we have kept them
            # at the end, so in case of a tie, it will take in order
            c_vote_dict = {90: 0, 270: 0, 0: 0, 180: 0}

            for root in tree_list:
                # static function
                pred_orientation = Forest.get_orientation_using_tree(root, image_array[i])
                c_vote_dict[pred_orientation] += 1

            final_pred_orientation = max(c_vote_dict, key=c_vote_dict.get)
            final_prediction_list.append(final_pred_orientation)
            if final_pred_orientation == true_orientation:
                total_correct += 1

        genrate_output(image_name, final_prediction_list)
        accuracy = (total_correct/total_set_size)*100

    print("Accuracy: {:.2f}%".format(accuracy))

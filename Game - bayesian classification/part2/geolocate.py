#!/usr/bin/env python3

from collections import Counter
from queue import PriorityQueue
from math import log
import operator
import sys


train, test, output = sys.argv[1:]

# http://xpo6.com/list-of-english-stop-words/
# stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]


# https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html
# stopwords = ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"]


# https://www.ranks.nl/stopwords : Google History
stopwords = ["about", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
             "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "who", "will", "with", "can"]

PROB_MISSING_WORD = 0.00005


def remove_stopwords(tweet):
    # removes all stopwords and single letter words
    return ' '.join([x for x in tweet.split() if x not in stopwords and len(x) > 1])


def cleaned(tweet):
    # remove capitalization
    tweet = tweet.lower()

    # remove characters which are not helpful in deciding the city
    chars_to_remove = '#@_,-()*.!:?+/0123456789\'\"~'
    for ch in chars_to_remove:
        tweet = tweet.replace(ch, ' ')

    tweet = remove_stopwords(tweet)
    return tweet


with open(train) as f:
    lines = f.readlines()


total_tweets = len(lines)

# stores word count by city
cities = dict()

# store number of tweets per city
tweets_per_city = dict()

# populated word counts per city and number of tweets per city
for i, l in enumerate(lines):
    # separate city from tweet
    idx = l.index(' ')
    city, tweet = l[:idx], cleaned(l[idx + 1:])

    if city not in cities:
        cities[city] = Counter()
        tweets_per_city[city] = 0

    cities[city] += Counter(tweet.split())
    tweets_per_city[city] += 1

# store count of words across all cities
count_words = Counter()

for city in cities:
    # filter words that occur less than 10 times
    cities[city] = {k: v for (k, v) in cities[city].items() if v >= 10}
    count_words += cities[city]

total_words = sum(count_words.values())
prob_word = {k: v/total_words for (k, v) in count_words.items()}


prob_word_given_city = {}
for city in cities:
    prob_word_given_city[city] = {k: v/sum(cities[city].values())
                                  for (k, v) in cities[city].items()}

# prior probability of tweet being from a city
prob_city = {k: v/total_tweets for (k, v) in tweets_per_city.items()}

# returns probability of city given a word using naive bayes theorem


def prob_city_given_word(city, word):
    if word not in prob_word_given_city[city]:
        prob_word_given_city[city][word] = PROB_MISSING_WORD
    return log(prob_word_given_city[city][word]) + log(prob_city[city]) - log(sum([(prob_word_given_city[x][word] if word in cities[x] else PROB_MISSING_WORD) * prob_city[x] for x in cities]))

# return n most significant words from tweet that help distinguishing between cities


def topn(tweet, n):
    tweet_words = cleaned(tweet).split()

    topn = []
    for city in cities:
        for word in tweet_words:
            pcw = prob_city_given_word(city, word)

            # take weighted probability as some words may occur few times in only 1 language
            if word in count_words:
                pcw += log(count_words[word])

            # if less than n words are selected, just add the current word
            if len(topn) < n:
                topn.append((word, pcw))
            else:
                # check if word already exists because of other city, then update the prob if greater than present
                found = False
                for i in range(len(topn)):
                    if word == topn[i][0]:
                        if pcw > topn[i][1]:
                            topn[i] = (word, pcw)
                        found = True
                        break
                if not found:
                    # replace with word with minimum weighted probability
                    minIndex = 0
                    for i in range(1, len(topn)):
                        if topn[i][1] < topn[minIndex][1]:
                            minIndex = i
                    if pcw > topn[minIndex][1]:
                        topn[minIndex] = (word, pcw)

    # return only words and not their probabilities
    return [x[0] for x in topn]


def get_city(tweet):
    words = topn(tweet, 7)

    max_prob = -float('inf')
    max_prob_city = None

    # calculate probability of city given list of words for each city and return the one with max probability
    for city in cities:
        # initialize prob with prior for city
        prob = log(prob_city[city])
        for word in words:
            if word in prob_word_given_city[city]:
                prob += log(prob_word_given_city[city][word])
            else:
                # for missing data assuming that word occurs with very low probablity
                prob += log(PROB_MISSING_WORD)

        if prob > max_prob:
            max_prob = prob
            max_prob_city = city

    return max_prob_city


out_file = []
correct, total = 0, 0
with open(test) as f:
    for l in f.readlines():
        city, tweet = l[:l.index(' ')].strip(), l[l.index(' ') + 1:].strip()
        estimated = get_city(tweet)
        if estimated == city:
            correct += 1
        total += 1
        out_file.append('{} {} {}'.format(estimated, city, tweet))
print('Accuracy: {:2f}%'.format(100*correct/total))

with open(output, 'w') as f:
    for line in out_file:
        f.write(line + '\n')


# returns n most significant words for given city
def topn_city(city, n):
    topn = []
    for word in prob_word_given_city[city]:
        pcw = prob_city_given_word(city, word)

        # if less than n words are selected, just add the current word
        if len(topn) < n:
            topn.append((word, pcw))
        else:
            # check if word already exists because of other city, then update the prob if greater than present
            found = False
            for i in range(len(topn)):
                if word == topn[i][0]:
                    if pcw > topn[i][1]:
                        topn[i] = (word, pcw)
                    found = True
                    break
            if not found:
                # replace with word with minimum probability
                minIndex = 0
                for i in range(1, len(topn)):
                    if topn[i][1] < topn[minIndex][1]:
                        minIndex = i
                if pcw > topn[minIndex][1]:
                    topn[minIndex] = (word, pcw)

    return [x[0] for x in sorted(topn, key=lambda x: x[1], reverse=True)]


print('Most distinguishing words per city')
for city in cities:
    print(city, topn_city(city, 5))

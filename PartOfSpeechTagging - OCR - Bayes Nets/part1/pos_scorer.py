###################################
# CS B551 Fall 2018, Assignment #3
# 
# Scoring code by D. Crandall
#
# PLEASE DON'T MODIFY THIS FILE.
# Edit pos_solver.py instead!
#

class Score:
    def __init__(self):
        self.word_scorecard = {}
        self.sentence_scorecard = {}
        self.word_count = 0
        self.sentence_count = 0


    def score(self, algo_outputs, gt):
        self.word_count += len(gt)
        self.sentence_count += 1

        for algo,labels in algo_outputs.items():
            correct = 0
            for j in range(0, len(gt)):
                correct += 1 if gt[j] == labels[j] else 0
        
            self.word_scorecard[algo] = self.word_scorecard.get(algo, 0) + correct
            self.sentence_scorecard[algo] = self.sentence_scorecard.get(algo, 0) + (correct == len(gt))


    def print_scores(self):
        print("\n==> So far scored %d sentences with %d words." % (self.sentence_count, self.word_count))
        print("                   Words correct:     Sentences correct: ")
        
        for i in sorted(self.word_scorecard):
            print("%18s:     %7.2f%%             %7.2f%%" % (i, self.word_scorecard[i]*100 / float(self.word_count), self.sentence_scorecard[i]*100 / float(self.sentence_count)))

    @staticmethod
    def print_helper(description, list, sentence):
        print (("%40s" % description) + " " + " ".join([(("%-" + str(max(4,len(sentence[i]))) + "s") % list[i]) for i in  range(0,len(list)) ] ) )

    @staticmethod
    def print_results(sentence, outputs, posteriors, models):
        Score.print_helper(" ".join([("%7s" % model) for model in models]), sentence, sentence)
        for algo in sorted(outputs.keys()):
            Score.print_helper(algo + "  "+" ".join(["%7.2f" % posteriors[algo][model] for model in models]), outputs[algo], sentence)


#Author: Ryan Liska
#Implementation of Viterbi Algorithm for POS Tagging

import numpy as np
import matplotlib.pyplot as plt
import math

class Viterbi_Algo():

    pos_count = {"##":0,
                 "$$":0,
                 ".":0,
                 "ADJ":0,
                 "ADP":0,
                 "ADV":0,
                 "CONJ":0,
                 "DET":0,
                 "NOUN":0,
                 "NUM":0,
                 "PRON":0,
                 "PRT":0,
                 "VERB":0,
                 "X":0}
    words = {}
    pattern_count = {}
    
    def __init__(self):
        for key in self.pos_count:
            self.pattern_count[key] = {"##":0,
                                       "$$":0,
                                       ".":0,
                                       "ADJ":0,
                                       "ADP":0,
                                       "ADV":0,
                                       "CONJ":0,
                                       "DET":0,
                                       "NOUN":0,
                                       "NUM":0,
                                       "PRON":0,
                                       "PRT":0,
                                       "VERB":0,
                                       "X":0}
    
    def parse_text(self, text):
        with open(text, "r") as file:
            for line in file:
                prev = "$$"
                for word in line.split():
                    word_list = word.rsplit("/", 1)
                    self.pos_count[word_list[1]] += 1
                    if prev != "$$":
                        self.pattern_count[prev][word_list[1]] += 1
                    if word_list[0] not in self.words.keys():
                        self.words[word_list[0]] = {"##":0,
                                                   "$$":0,
                                                   ".":0,
                                                   "ADJ":0,
                                                   "ADP":0,
                                                   "ADV":0,
                                                   "CONJ":0,
                                                   "DET":0,
                                                   "NOUN":0,
                                                   "NUM":0,
                                                   "PRON":0,
                                                   "PRT":0,
                                                   "VERB":0,
                                                   "X":0}
                    self.words[word_list[0]][word_list[1]] += 1
                    prev = word_list[1]
                    
    def visual1(self):
        words_dict = {"family":0, "guy":0, "peter":0, "griffin":0}
        word = []
        values = []
        for key in words_dict.keys():
            word.append(key)
            for value in self.words[key].values():
                words_dict[key] += value
            values.append(words_dict[key])
        fig = plt.figure(figsize = (10, 5))
        plt.bar(word, values, width = 0.4)
        plt.ylabel("Counts")
        for index, value in enumerate(values):
            plt.text(index, value, str(value))
        plt.show()
        
    def visual2(self):
        list_of_pos = []
        for key in self.pos_count.keys():
            if key != "##" and key != "$$":
                list_of_pos.append((self.pos_count[key], key))
        sorted_list = sorted(list_of_pos, reverse = True)
        values = []
        pos = []
        for i in sorted_list:
            values += [i[0]]
            pos += [i[1]]
        fig = plt.figure(figsize = (10, 5))
        plt.xlabel("Counts")
        plt.barh(range(len(values)), values)
        plt.yticks(range(len(values)), pos);
        for index, value in enumerate(values):
            plt.text(value, index, str(value))
        plt.show()
        
    def score_estimation(self, pos1, word, pos2):
        emission = (self.words[word][pos2] + 1) / (self.pos_count[pos2] + 12)
        transition = (self.pattern_count[pos1][pos2] + 1) / (self.pos_count[pos1] + 12)
        return math.log(emission, 10) + math.log(transition, 10)
    
    def algo_end(self, word, prev_scores):
        best_score = 0
        best_pos = ""
        for key in self.pos_count.keys():
            score = self.score_estimation(key, word, "$$") + prev_scores[key]
            if best_score == 0 or score > best_score:
                best_score = score
                best_pos = key
        return [best_pos]
    
    def algo_help(self, word_list, prev_scores):
        scores = {}
        prev_pos = {}
        for key1 in self.pos_count.keys():
            best_score = 0
            best_pos = ""
            for key2 in self.pos_count.keys():
                score = self.score_estimation(key2, word_list[0], key1) + prev_scores[key2]
                if best_score == 0 or score > best_score:
                    best_score = score
                    best_pos = key2
            scores[key1] = best_score
            prev_pos[key1] = best_pos
        if len(word_list) == 1:
            pos_list = self.algo_end(word_list[0], scores)
        else:
            pos_list = self.algo_help(word_list[1:], scores)
        pos_list.insert(0, prev_pos[pos_list[0]])
        return pos_list
                
        
    def algo(self, sentance):
        word_list = sentance.split()
        scores = {}
        for key in self.pos_count.keys():
            scores[key] = self.score_estimation("##", word_list[0], key)
        return self.algo_help(word_list[1:], scores)
    
    def test(self):
        with open("test.txt", "r") as file:
            for line in file:
                print(line)
                print(self.algo(line))
                print()


if __name__ == '__main__':
    va = Viterbi_Algo()
    va.parse_text("tagged_sentences.txt")
    va.visual1()
    va.visual2()
    va.test()
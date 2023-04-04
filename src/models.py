'''
    Implemenation of Language Models 
'''

import nltk
from preprocess import Preprocess


# N-Gram Language Model
class NGram:
    def __init__(self, tokens, n):
        self.n = n
        self.tokens = tokens
        self.model = self.buildModel(tokens, n)
    
    def buildModel(self, tokens, n):
        nGram = nltk.ngrams(tokens, n)
        nGramCount = nltk.FreqDist(nGram)

        if n == 1:
            return {v: c/len(nGramCount) for v, c in nGramCount.items()}
        else:
            mGrams = nltk.ngrams(tokens, n-1)
            mGramCounts = nltk.FreqDist(mGrams)

            def nGramProb(nGram, nGramCount):
                mGram = nGram[:-1]
                mGramCount = mGramCounts[mGram]
                return nGramCount/mGramCount
            return {v: nGramProb(v, c) for v, c in nGramCount.items()}
        
# Word2Vec Language Model
class Word2Vec():
    def __init__(self, tokens):
        self.tokens = tokens
        self.model = self.buildModel(tokens)
    
    def buildModel(self, tokens):
        something


if __name__ == "__main__":

    ### Hyperparameters ###
    n = 2 # n-gram
    #######################

    # sentences = ["Unsurprisingly, the development of humanoid robots have been very much a part of the development of legged robots in general, as platforms such as quadrupeds still share many of the similar core technologies despite their silhouettes having distinct differences. This is because they face similar challenges mentioned in Section 1.1"]
    # sentences = ['She sells seashells on the seashore.',"The shells she sells are seashells I'm sure","So if she sells seashells on the seashore.", "I'm sure that the shells are seashore shells.", "For if she sells seashells on the seashore."]
    sentences = ['''Machine learning is the study of computer algorithms that \
                improve automatically through experience. It is seen as a \
                subset of artificial intelligence. Machine learning algorithms \
                build a mathematical model based on sample data, known as \
                training data, in order to make predictions or decisions without \
                being explicitly programmed to do so. Machine learning algorithms \
                are used in a wide variety of applications, such as email filtering \
                and computer vision, where it is difficult or infeasible to develop \
                conventional algorithms to perform the needed tasks.''']
    tokens = Preprocess(sentences).Run()
    
    nGram = NGram(tokens, n)
    print(f"Tokens: {nGram.tokens}")
    print("\n")
    print(f"Model: {nGram.model}")
    
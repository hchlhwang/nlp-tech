'''
    Preprocess language data
'''
import time
import nltk
import re
from functools import reduce

class Preprocess:
    def __init__(self, sentences):
        self.sentences = sentences

    # Lowercase all words
    def lowerCase(self):
        return list(map(str.lower, self.sentences))

    # Add <s> and </s> to the beginning and end of each sentence
    def addBosEos(self, n=2):
        BOS = '<s>'
        EOS = '</s>'
        BOSs = ' '.join([BOS] * (n-1) if n>1 else [BOS])
        sentences = [' '.join([BOSs, sentence, EOS]) for sentence in self.sentences]
        return sentences
        # return list(map(lambda x: '<s> ' + x + ' </s>', sentences))

    # Tokenize sentences
    def tokenize(self):
        sentences = list(map(lambda s: s.split(' '), self.sentences))
        tokens = list(reduce(lambda a, b: a+b, sentences))
        return tokens

    # Replace words that appear only once with <unk>
    def onceToUnk(self, tokens):
        UNK = '<unk>'
        freq = nltk.FreqDist(tokens)
        tokens = [t if freq[t] > 1 else UNK for t in tokens]
        return tokens
    
    # Remove punctuation (token should be a string)
    def removePunctuation(self, tokens):
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        return pattern.findall(tokens.lower())

    # Preprocess sentences
    def run(self):
        self.sentences = self.lowerCase()
        self.sentences = self.addBosEos()
        tokens = self.tokenize()
        tokens = self.onceToUnk(tokens)
        return tokens

    # Map words to integers
    def mapping(self, tokens):
        word2idx, idx2word = {}, {}

        for i, token in enumerate(set(tokens)): # set() removes duplicates
            word2idx[token] = i
            idx2word[i] = token
        
        return word2idx, idx2word

def main():
    sentences = ['She sells seashells on the seashore.',"The shells she sells are seashells I'm sure","So if she sells seashells on the seashore.",\
                      "I'm sure that the shells are seashore shells.", "For if she sells seashells on the seashore."]
    sentences = ["Machine learning is the study of computer algorithms that \
                    improve automatically through experience. It is seen as a \
                    subset of artificial intelligence. Machine learning algorithms \
                    build a mathematical model based on sample data, known as \
                    training data, in order to make predictions or decisions without \
                    being explicitly programmed to do so. Machine learning algorithms \
                    are used in a wide variety of applications, such as email filtering \
                    and computer vision, where it is difficult or infeasible to develop \
                    conventional algorithms to perform the needed tasks."]
    sentences2 = '''Machine learning is the study of computer algorithms that \
                improve automatically through experience. It is seen as a \
                subset of artificial intelligence. Machine learning algorithms \
                build a mathematical model based on sample data, known as \
                training data, in order to make predictions or decisions without \
                being explicitly programmed to do so. Machine learning algorithms \
                are used in a wide variety of applications, such as email filtering \
                and computer vision, where it is difficult or infeasible to develop \
                conventional algorithms to perform the needed tasks.'''
    p = Preprocess(sentences)
    preprocessedTokens = p.run()
    # print(preprocessedTokens)

    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    # print(pattern.findall(sentences2.lower()))

    w2i, i2w = p.mapping(preprocessedTokens)
    # print(w2i)
    # print(i2w)


if __name__ == "__main__":
    main()
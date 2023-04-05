'''
    Dataset class for the GRU dataset
'''

import sys
import torch
import nltk
from functools import reduce
from torch.utils.data import Dataset

    
def gruPreprocess(sentences, addSpecialTokens=True):
    # Lowercase
    sentences = list(map(str.lower, sentences))

    # Add BOS and EOS
    if addSpecialTokens:
        BOS = '<s>'
        EOS = '</s>'
        sentences = [' '.join([BOS, sentence, EOS]) for sentence in sentences]
    
    # Tokenize
    sentences = list(map(lambda s: s.split(' '), sentences))

    return sentences

class GRUModelDataset(Dataset):
    '''
        Dataset class for the GRU language model

        Args:
            text: entire text dataset
        Returns:
            Dataset class (tokenized text -> tensor object)
    '''
    def __init__(self, text):
        sentenceList = nltk.tokenize.sent_tokenize(text)
        tokenizedSentences = gruPreprocess(sentenceList)
        tokens = list(reduce(lambda a, b: a+b, tokenizedSentences))
        self.vocab = self.makeVocab(tokens)
        self.i2v = {v:k for k, v in self.vocab.items()}
        self.indice = list(map(lambda s: self.convertTokensToIndices(s), tokenizedSentences))

    def convertTokensToIndices(self, sentence):
        indice = []
        for s in sentence:
            try:
                indice.append(self.vocab[s])
            except KeyError:
                indice.append(self.vocab['<unk>'])
        return torch.tensor(indice)
    
    def makeVocab(self, tokens):
        vocab = {}
        vocab['<pad>'] = 0
        vocab['<s>'] = 1
        vocab['</s>'] = 2
        vocab['<unk>'] = 3
        index = 4

        for t in tokens:
            try:
                vocab[t]
                continue
            except KeyError:
                vocab[t] = index
                index += 1
        return vocab

    def __len__(self):
        return len(self.indice)
    
    def __getitem__(self, idx):
        return self.indice[idx]
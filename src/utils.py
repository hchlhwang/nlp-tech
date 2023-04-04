import nltk

def LaplaceSmoothing(ngram, ngramCount, ngramMinusOneCount, vocabSize):
    return (ngramCount + 1) / (ngramMinusOneCount + vocabSize)

# Add-k smoothing
def AddKSmoothing(ngram, ngramCount, k): # Laplace smoothing with k = 1
    return (ngramCount + k) / (ngramCount + k * len(ngramCount))
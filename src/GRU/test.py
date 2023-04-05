from dataset import GRUModelDataset
import nltk
nltk.download('punkt')

text = 'she sells sea shells by the sea shore'
dataset = GRUModelDataset(text)

for d in dataset:
    print(d)
    break

print(dataset.vocab)
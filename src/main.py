from models import NGram
from preprocess import Preprocess


def main():
    sentences = ['She sells seashells on the seashore.',"The shells she sells are seashells I'm sure","So if she sells seashells on the seashore.",\
            "I'm sure that the shells are seashore shells.", "For if she sells seashells on the seashore."]
    tokens = Preprocess(sentences).Run()
    nGram = NGram(tokens, 2)
    print(f"Tokens: {nGram.tokens}")
    print("\n")
    print(f"Model: {nGram.model}")
    
if __name__ == "__main__":
    main()
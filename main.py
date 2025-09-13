from src.probability_generator import ProbabilityGenerator
from src.text_embedder import TextEmbedder
from src.text_generator import TextGenerator


def main():
    e = TextEmbedder()
    embeddings = e.create_embeddings()
    vocab = e.vocabulary()
    p = ProbabilityGenerator(embeddings)
    probability = p.probability_matrix()
    g = TextGenerator(
        probability=probability,
        vocabulary=vocab,
    )
    generated = g.generate()

    with open("output/generated.txt", "w") as f:
        f.write(generated)


if __name__ == "__main__":
    main()

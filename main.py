from src.probability_generator import ProbabilityGenerator
from src.text_embedder import TextEmbedder
from src.text_generator import TextGenerator


def main():
    with open("datasets/0.dataset") as f:
        content = f.read()

    g = TextGenerator()
    generated = g.generate(content)

    with open("output/generated.txt", "w") as f:
        f.write(generated)


if __name__ == "__main__":
    main()

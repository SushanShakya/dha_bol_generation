from src.embeddings_generator import EmbeddingsGenerator
from src.nn import NN
from src.probability_generator import ProbabilityGenerator
from src.text_embedder import TextEmbedder


class TextGenerator:

    def generate(self, sample):
        vocab = list(set(sample.split(" ")))

        e = TextEmbedder(vocab)
        embeddings = e.embed(sample)

        p = ProbabilityGenerator(embeddings)
        probability = p.probability_matrix()

        g = EmbeddingsGenerator(probability)
        generated = g.generate()

        return " ".join(e.unembed(generated))

    def generate_from_nn(self, sample):
        vocab = list(set(sample.split(" ")))

        e = TextEmbedder(vocab)
        embeddings = e.embed(sample)

        bigram = list(zip(embeddings, embeddings[1:]))

        x = list(map(lambda a: a[0], bigram))
        y = list(map(lambda a: a[1], bigram))

        nn = NN(vocab)

        for _ in range(1000):
            loss = nn.gradient_descent(x, y)
            # print(loss)

        print(e.itos)

        generated = nn.random_walk()

        return " ".join(e.unembed(generated))

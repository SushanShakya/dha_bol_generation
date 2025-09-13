from src.embeddings_generator import EmbeddingsGenerator
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

from src.embeddings_generator import EmbeddingsGenerator
from src.probability_generator import ProbabilityGenerator
from src.text_embedder import TextEmbedder
from src.vocab_generator import VocabGenerator


class TextGenerator:

    sample: str

    def __init__(self, sample: str):
        self.sample = sample

    def _generate_vocab(self):
        return VocabGenerator().generate(self.sample)

    def generate(self):
        vocab = self._generate_vocab()

        e = TextEmbedder(vocab)
        embeddings = e.embed(self.sample)

        p = ProbabilityGenerator(embeddings)
        probability = p.probability_matrix()

        g = EmbeddingsGenerator(probability)
        generated = g.generate()

        return " ".join(e.unembed(generated))

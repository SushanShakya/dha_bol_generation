from unittest import TestCase

from src.vocab_generator import VocabGenerator


class TestVocabGenerator(TestCase):

    def setUp(self):
        self.g = VocabGenerator()

    def test_clean_text(self):
        inputs = [
            "a b c\na b c",
            "a b c\n\na b c",
            "a b c\n\n\na b c",
            "a b c\n\n\n\na b c",
        ]
        output = "a b c a b c"
        for i in inputs:
            cleaned = self.g.clean_text(i)
            self.assertEqual(cleaned, output)

    def test_generate(self):
        text = "धाँ दि त त\nघे घे ताक घे"
        result = self.g.generate(text)

        self.assertEqual(len(result), len(set(result)))

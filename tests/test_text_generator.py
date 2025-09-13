from unittest import TestCase

from src.text_generator import TextGenerator


class TestTextGenerator(TestCase):
    def test_generate(self):
        sample = "रिक त ख ति धाँ"
        g = TextGenerator(sample)
        generated = g.generate()
        self.assertEqual(generated, sample)

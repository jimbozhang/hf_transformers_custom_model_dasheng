import unittest

import torch
import torchaudio
from dasheng_model.feature_extraction_dasheng import DashengFeatureExtractor
from dasheng_model.modeling_dasheng import DashengModel


class TestInference(unittest.TestCase):

    def test_dasheng_for_audio_classification(self):
        model_id = "mispeech/dasheng-base"
        feature_extractor = DashengFeatureExtractor.from_pretrained(model_id)
        model = DashengModel.from_pretrained(model_id)

        audio, sampling_rate = torchaudio.load("resources/JeD5V5aaaoI_931_932.wav")

        inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        assert logits.shape == (1, 768)


if __name__ == "__main__":
    """
    Run the tests:
        python -m unittest tests/*.py
    """

    unittest.main()

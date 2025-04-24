import unittest

import torch
import torchaudio
from dasheng_model.feature_extraction_dasheng import DashengFeatureExtractor
from dasheng_model.modeling_dasheng import DashengModel


class TestInference(unittest.TestCase):

    def setUp(self):
        model_id = "mispeech/dasheng-base"
        self.feature_extractor = DashengFeatureExtractor.from_pretrained(model_id)
        self.model = DashengModel.from_pretrained(model_id)
        self.audio, self.sampling_rate = torchaudio.load("resources/JeD5V5aaaoI_931_932.wav")


    def test_dasheng_for_audio_classification(self):

        inputs = self.feature_extractor(self.audio, sampling_rate=self.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        assert logits.shape == (1, 768)

    def test_dasheng_for_frame_level_feature_extraction(self):
        inputs = self.feature_extractor(self.audio, sampling_rate=self.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            frame_level_hiddens = self.model(**inputs).hidden_states

        assert frame_level_hiddens.shape == (1, 25, 768)

    def test_dasheng_for_frame_level_feature_extraction_long_padded(self):
        #pad 10s
        padded_audio = torch.nn.functional.pad(self.audio, (0, 160000))

        inputs = self.feature_extractor(padded_audio, sampling_rate=self.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            frame_level_hiddens = self.model(**inputs).hidden_states

        assert frame_level_hiddens.shape == (1, 275, 768)

if __name__ == "__main__":
    """
    Run the tests:
        python -m unittest tests/*.py
    """

    unittest.main()

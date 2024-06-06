import unittest

import numpy as np
import torch

from dasheng_model.feature_extraction_dasheng import DashengFeatureExtractor


class TestDashengFeatureExtractor(unittest.TestCase):

    def test_extract_feature_from_pt_tensor(self):
        feature_extractor = DashengFeatureExtractor()
        x = torch.randn(2, 16000)
        y = feature_extractor(x)
        assert y["input_values"].shape == (2, 64, 101)

    def test_extract_feature_from_np_array(self):
        feature_extractor = DashengFeatureExtractor()
        x = np.random.rand(2, 16000)
        y = feature_extractor(x)
        assert y["input_values"].shape == (2, 64, 101)

    def test_extract_feature_from_list_of_pt_tensors(self):
        feature_extractor = DashengFeatureExtractor()
        x = [torch.randn(16000), torch.randn(16000)]
        y = feature_extractor(x)
        assert y["input_values"].shape == (2, 64, 101)

    def test_extract_feature_from_list_of_np_arrays(self):
        feature_extractor = DashengFeatureExtractor()
        x = [np.random.rand(16000), np.random.rand(16000)]
        y = feature_extractor(x)
        assert y["input_values"].shape == (2, 64, 101)

    def test_extract_feature_from_list_of_np_arrays_pad(self):
        feature_extractor = DashengFeatureExtractor()
        x = [np.random.rand(10), np.random.rand(16000)]
        y = feature_extractor(x, max_length=32000)
        assert y["input_values"].shape == (2, 64, 201)

    def test_extract_feature_from_list_of_np_arrays_pad_trim(self):
        feature_extractor = DashengFeatureExtractor()
        x = [np.random.rand(10), np.random.rand(16000)]
        y = feature_extractor(x, max_length=16000)
        assert y["input_values"].shape == (2, 64, 101)


if __name__ == "__main__":
    """
    Run the tests:
        python -m unittest tests/*.py
    """

    unittest.main()

import pytest
import os
from pathlib import Path
from production.data import CustomPreprocessor

@pytest.fixture
def dataset_loc():
    return os.path.join(Path.home(),'intent_recognition/datasets','search_queries_candidate_dataset__jan_2024.csv')

@pytest.fixture
def preprocessor():
    return CustomPreprocessor
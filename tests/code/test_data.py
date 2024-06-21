import pandas as pd
import pytest
import ray

from production import data

@pytest.fixture(scope='module')
def df():
    data=[{'date':'2024:06:19',
           'search_query':'übersetzen tomatoes',
           'market':'de-de',
           'geo_country':'DE',
           'device_type':'Mobile',
           'browser_name':'Safari',
           'intent':'TRANSLATION',
           'daily_query_count':2}]
    df = pd.DataFrame(data)
    return df

@pytest.fixture(scope='module')
def class_to_index():
    class_to_index={'TRANSLATION':0,'WEATHER':1}
    return class_to_index

def test_load_data(dataset_loc):
    num_samples = 10
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=num_samples)
    assert ds.count() == num_samples

def test_stratify_split():
    n_per_class = 2
    intents = n_per_class * ["TRANSLATION"] + n_per_class * ["WEATHER"]
    ds = ray.data.from_items([dict(intent=t) for t in intents])
    train_ds, test_ds = data.stratify_split(ds, stratify="intent", test_size=0.5)
    train_target_counts = train_ds.to_pandas().target.value_counts().to_dict()
    test_target_counts = test_ds.to_pandas().target.value_counts().to_dict()
    assert train_target_counts == test_target_counts

def test_preprocess(df, class_to_index):
    #assert "text" not in df.columns
    outputs = data.preprocess(df, class_to_index=class_to_index)
    assert set(outputs) == {"ids", "masks", "targets"}


def test_fit_transform(dataset_loc, preprocessor):
    ds = data.load_data(dataset_loc=dataset_loc)
    preprocessor = preprocessor.fit(ds)
    preprocessed_ds = preprocessor.transform(ds,'test')
    assert len(preprocessor.class_to_index) == 8
    assert ds.count() == preprocessed_ds.count()

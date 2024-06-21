import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def load_data(dataset_loc: str) -> Dataset:
    """Load data from source into a Ray Dataset. 

    Args:
        dataset_loc (str): Location of the dataset.

    Returns:
        Dataset: Our dataset represented by a Ray Dataset.
    """
    df = pd.read_csv(dataset_loc)
    df=df.loc[~df['intent'].isnull()]
    ds = ray.data.from_pandas(df)
    ds = ds.random_shuffle(seed=1234)
    #ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds

def stratify_split(
    ds: Dataset,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split a dataset into train and test splits with equal
    amounts of data points from each class in the column we
    want to stratify on.

    Args:
        ds (Dataset): Input dataset to split.
        stratify (str): Name of column to split on.
        test_size (float): Proportion of dataset to split for test set.
        shuffle (bool, optional): whether to shuffle the dataset. Defaults to True.
        seed (int, optional): seed for shuffling. Defaults to 1234.

    Returns:
        Tuple[Dataset, Dataset]: the stratified train and test datasets.
    """

    def _add_split(df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover, used in parent function
        """Naively split a dataframe into train and test splits.
        Add a column specifying whether it's the train or test split."""
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:  # pragma: no cover, used in parent function
        """Filter by data points that match the split column's value
        and return the dataframe with the _split column dropped."""
        return df[df["_split"] == split].drop("_split", axis=1)

    # Train, test split with stratify
    grouped = ds.groupby(stratify).map_groups(_add_split, batch_format="pandas")  # group by each unique value in the column we want to stratify on
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "train"}, batch_format="pandas")  # combine
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "test"}, batch_format="pandas")  # combine

    # Shuffle each split (required)
    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds

def combine_text(df,mode):
    # This will hold all of the dataset samples, as strings.
    combined_text = []
    labels = []

    print('Combining features into strings...')

    # For each of the samples...
    for index, row in df.iterrows():    
        combined = ""

        combined += "This search query is {:}. It comes from the {:} market and {:} country. " \
                    "The device used is {:} and the browser is {:} with a daily query count of {:}. ".format(row["search_query"], 
                                                           row["market"], 
                                                           row["geo_country"],
                                                           row["device_type"],
                                                           row["browser_name"],
                                                           row["daily_query_count"])


        # Add the combined text to the list.
        combined_text.append(combined)

        
        if mode!='infer':
            # Collect the sample's label.
            labels.append(row["intent"]) 
        elif mode=='infer':
            labels.append(-1)       

    print('  DONE.')

    print('Dataset contains {:,} samples.'.format(len(combined_text)))
    return combined_text,labels
    
def tokenize(queries, labels):
    huggingface_model='bert-base-multilingual-uncased'
    tokenizer=AutoTokenizer.from_pretrained(huggingface_model,do_lower_case=True)
    encoded_dict = tokenizer(queries, return_tensors="np", padding="longest")
    #encoded_dict=tokenizer.batch_encode_plus(queries,
    #                           add_special_tokens=True,
    #                          max_length = 60,           # Pad & truncate all sentences.
    #                            truncation = True,
    #                            padding = 'max_length',
    #                            return_attention_mask=True,
    #                            return_tensors='np'
    #                            )
    #dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))   
    return {'ids':encoded_dict['input_ids'],'masks':encoded_dict['attention_mask'],'targets':np.array(labels)}

def preprocess(df: pd.DataFrame, intent_to_index: Dict, mode: str) -> Dict:
    """Preprocess the data in our dataframe.

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.
        intent_to_index (Dict): Mapping of class names to indices.

    Returns:
        Dict: preprocessed data (ids, masks, targets).
    """ 
    #1. drop the date metric        
    df.drop('date',axis=1,inplace=True)  
    
    #2. fill missing values of categorical metrics to 'other'
    for metric in ['market','geo_country','device_type','browser_name']: 
        if df[metric].isnull().any():
            df[metric]=df[metric].fillna('other')

    #3. fill missing values of numeric metrics to '0'
    if df['daily_query_count'].isnull().any():
        df['daily_query_count']=df['daily_query_count'].fillna(0)

    #4. label encoding
    if mode!='infer':
        df["intent"] = df["intent"].map(intent_to_index)  

    #5. combine all metrics into a single text input and tokenize
    queries,labels = combine_text(df,mode)
    outputs = tokenize(queries,labels)
    return outputs      


class CustomPreprocessor:
    """Custom preprocessor class."""

    def __init__(self, intent_to_index={}):
        self.intent_to_index = intent_to_index or {}  # mutable defaults
        self.index_to_intent = {v: k for k, v in self.intent_to_index.items()}
        
    def fit(self, ds):
        tags = ds.unique(column="intent")
        print(tags)
        self.intent_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_intent = {v:k for k, v in self.intent_to_index.items()}
        return self
    
    def transform(self, ds, mode):
        return ds.map_batches(
            preprocess, 
            fn_kwargs={"intent_to_index": self.intent_to_index, "mode":mode}, 
            batch_format="pandas")

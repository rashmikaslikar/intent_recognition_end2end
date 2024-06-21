import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from ray.train.torch import get_device

from production import utils

def test_set_seed():
    utils.set_seeds()
    a=np.random.randn(2,3)
    b=np.random.randn(2,3)
    utils.set_seeds()
    x=np.random.randn(2,3)
    y=np.random.randn(2,3)

    assert np.array_equal(a,x)
    assert np.array_equal(b,y)

def test_save_load_dict():
    with tempfile.TemporaryDirectory as fp:
        d = {'roku':'kyoshi'}
        path = Path(fp,'d.json')
        utils.save_dict(d=d, path=path)
        d = utils.load_dict(path=path)
        assert d['roku'] == 'kyoshi'

def test_pad_array():
    array = np.array([[1,2],[1,2,3]],dtype='object')
    padded_array = np.array([[1,2,0],[1,2,3]])
    assert np.array_equal(utils.pad_array(array),padded_array)

def test_collate_fn():
    batch = {
            'ids':np.array([[1,2],[1,2,3]],dtype='object'),
            'masks':np.array([[1,1],[1,1,1]],dtype='object'),
            'targets':np.array([1,2])
    }
    processed_batch = utils.collate_fn(batch)
    expected_batch = {
            'ids':np.array([[1,2,0],[1,2,3]],dtype=torch.int32, device='cpu'),
            'masks':np.array([[1,1,0],[1,1,1]],dtype=torch.int32, device='cpu'),
            'targets':np.array([1,2])
    }

    for k in batch:
        torch.allclose(processed_batch[k],expected_batch[k])

@pytest.mark.parametrize(
'd, keys, list',
[
 ({'a':[1,2],'b':[2,3]},['a','b'],[{'a':1,'b':2},{'a':2,'b':3}]),   
 ({'a':[1,2],'b':[2,3]},['a'],[{'a':1},{'a':2}]),
],

)
def test_dict_to_list(d,keys,list):
    assert utils.dict_to_list(d,keys)==list

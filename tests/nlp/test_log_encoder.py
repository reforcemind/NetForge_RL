import pytest
import numpy as np
from netforge_rl.nlp.log_encoder import LogEncoder, EMBEDDING_DIM


@pytest.fixture
def encoder():
    return LogEncoder(backend='tfidf')


@pytest.mark.fast
def test_encoder_single_line(encoder):
    log = '4624 - Success Logon by SYSTEM from 192.168.1.5'
    vec = encoder.encode(log)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EMBEDDING_DIM,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-05)


@pytest.mark.fast
def test_encoder_empty_input(encoder):
    vec = encoder.encode('')
    assert np.allclose(vec, 0.0)
    vec_none = encoder.encode(None)
    assert np.allclose(vec_none, 0.0)


@pytest.mark.fast
def test_encoder_buffer_aggregation(encoder):
    logs = [
        '4624 - Success Logon',
        'Sysmon 3 - Network Connection',
        '4688 - Process Created',
    ]
    vec_mean = encoder.encode_buffer(logs, agg='mean')
    assert vec_mean.shape == (EMBEDDING_DIM,)
    vec_max = encoder.encode_buffer(logs, agg='max')
    assert vec_max.shape == (EMBEDDING_DIM,)
    assert not np.allclose(vec_mean, vec_max)


@pytest.mark.fast
def test_encoder_caching(encoder):
    log = 'Repeated Log Line for Cache Test'
    vec1 = encoder.encode(log)
    vec2 = encoder.encode(log)
    assert np.array_equal(vec1, vec2)

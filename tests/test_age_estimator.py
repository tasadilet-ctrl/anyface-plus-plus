"""Tests for the age head: the network shape, the age codes, and the estimator.

Two bugs are guarded here, both of which made the age branch of the pipeline
unrunnable rather than merely inaccurate:

  * _AgeNet used one ``channels`` argument as both the input and the output
    width of a stage, so the first forward pass died on a shape mismatch.
  * age_estimator.py called cv2 without importing it, so every estimate()
    raised NameError before reaching the network at all.

The rest covers the age codes, which decide what a single wrong output unit
costs in years. These run on CPU in a second and need no weights or images.
"""
import numpy as np
import pytest
import torch

from anyface_pp.models.age_estimator import (ENCODINGS, MAX_AGE, NUM_BITS,
                                             AgeEstimator, _AgeNet, decode_age,
                                             encode_age, num_outputs)

ALL_AGES = np.arange(MAX_AGE + 1)


def _crop(value=128, h=48, w=40):
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------- network
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_forward_pass_runs_and_has_one_output_per_code_unit(encoding):
    """The stage widths must actually line up: 64 -> 128 -> 256 -> 512."""
    net = _AgeNet(encoding=encoding).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, num_outputs(encoding))


def test_forward_pass_runs_with_the_full_resolution_stem():
    net = _AgeNet(downsample_stem=False).eval()
    with torch.no_grad():
        assert net(torch.randn(1, 3, 224, 224)).shape == (1, MAX_AGE)


def test_network_returns_logits_not_probabilities():
    """Training uses BCEWithLogitsLoss, which must not be fed sigmoids."""
    net = _AgeNet().eval()
    with torch.no_grad():
        out = net(torch.randn(8, 3, 224, 224))
    assert out.min() < 0.0, "outputs are all positive; a sigmoid may have crept back in"


# ------------------------------------------------------------------ codes
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_encode_decode_round_trips_exactly(encoding):
    assert (decode_age(encode_age(ALL_AGES, encoding), encoding) == ALL_AGES).all()


def test_ordinal_code_is_a_thermometer():
    """Unit k means 'older than k', so the age is just the number of ones."""
    codes = encode_age(np.array([0, 1, 40, MAX_AGE]), "ordinal")
    assert codes.sum(1).tolist() == [0, 1, 40, MAX_AGE]
    # every row is a run of ones followed by a run of zeros
    for row in codes:
        assert np.all(np.diff(row) <= 0)


def test_binary_code_is_place_valued_msb_first():
    assert encode_age(np.array([1]), "binary")[0].tolist() == [0] * 7 + [1]
    assert encode_age(np.array([128]), "binary")[0].tolist() == [1] + [0] * 7


@pytest.mark.parametrize("encoding,expected", [("ordinal", 1.0), ("binary", 128.0)])
def test_cost_of_a_single_wrong_unit(encoding, expected):
    """The whole reason ordinal is the default: one wrong unit, one year."""
    codes = encode_age(ALL_AGES, encoding)
    worst = 0.0
    for unit in range(codes.shape[1]):
        flipped = codes.copy()
        flipped[:, unit] = 1.0 - flipped[:, unit]
        worst = max(worst, float(np.abs(decode_age(flipped, encoding) - ALL_AGES).max()))
    assert worst == expected


def test_ordinal_targets_are_monotone_but_binary_targets_alternate():
    """How many times a unit's target flips across the age range.

    Every ordinal unit is one step. The last binary bit is age parity, which
    no face reveals.
    """
    def flips(encoding):
        codes = encode_age(ALL_AGES, encoding)
        return np.abs(np.diff(codes, axis=0)).sum(0).astype(int)

    assert set(flips("ordinal").tolist()) == {1}
    binary = flips("binary").tolist()
    assert binary[-1] == MAX_AGE          # weight-1 bit: flips every year
    assert binary[0] == 0                 # weight-128 bit: dead over 0..100


def test_ordinal_decoding_survives_a_non_monotone_prediction():
    """Units are predicted independently, so the run of ones can have holes.

    Reading off the first zero would truncate to 3; counting gives 40.
    """
    probs = np.zeros((1, MAX_AGE), dtype=np.float32)
    probs[0, :41] = 0.9
    probs[0, 3] = 0.1                     # one dropped unit in the middle
    assert decode_age(probs, "ordinal")[0] == 40


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="encoding must be one of"):
        encode_age(np.array([30]), "onehot")


# -------------------------------------------------------------- estimator
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_estimate_returns_a_number_rather_than_raising(encoding):
    """Regression test for the missing cv2 import: this used to be NameError."""
    est = AgeEstimator(encoding=encoding)
    age = est.estimate(_crop())
    assert isinstance(age, float)
    assert 0 <= age <= 2 ** NUM_BITS - 1


def test_estimate_batch_matches_estimate_and_spans_batches():
    est = AgeEstimator()
    crops = [_crop(v) for v in (0, 90, 180, 255)]
    batched = est.estimate_batch(crops, batch_size=2)   # forces two batches
    assert batched == [est.estimate(c) for c in crops]


def test_checkpoint_encoding_wins_over_the_constructor_argument(tmp_path):
    """A binary model decoded as ordinal would return quiet nonsense."""
    path = tmp_path / "age.pth"
    AgeEstimator(encoding="binary").save_weights(path)
    loaded = AgeEstimator(weights_path=path, encoding="ordinal")
    assert loaded.encoding == "binary"


def test_missing_weights_file_is_an_error_not_a_random_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        AgeEstimator(weights_path=tmp_path / "does_not_exist.pth")

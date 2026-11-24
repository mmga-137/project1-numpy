"""Reproduce a patch-order mistake on synthetic pixels; no course data needed.

This is an explicit debugging experiment, not a record of an accidental bug.
Run from the repository root: python docs/debug_patch_order.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_encoder import ImageEncoder


def main():
    images = np.arange(3 * 224 * 224, dtype=np.float32).reshape(1, 3, 224, 224)
    expected = np.stack([
        images[0, :, top:top + 32, left:left + 32].flatten()
        for top in range(0, 224, 32)
        for left in range(0, 224, 32)
    ])[None]
    wrong = images.reshape(1, 49, 3072)
    corrected = ImageEncoder.__new__(ImageEncoder).patchify(images)
    print('Experiment: direct reshape has the right shape but the wrong pixel order.')
    print('All shapes:', expected.shape, wrong.shape, corrected.shape)
    print('Wrong first patch, first 36 values:', wrong[0, 0, :36].tolist())
    print('Correct first patch, first 36 values:', corrected[0, 0, :36].tolist())
    print('Wrong reshape mismatches:', np.count_nonzero(wrong != expected))
    print('Corrected patchify mismatches:', np.count_nonzero(corrected != expected))
    assert not np.array_equal(wrong, expected)
    np.testing.assert_array_equal(corrected, expected)


if __name__ == '__main__':
    main()

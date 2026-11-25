"""Export stage tolerances and selected actual/teacher pairs for the report.

Run from the repository root:
    python tests/compare_reference.py --output results/reference_comparison.json
Teacher arrays are read only by this validation script, never by the encoder.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check import verify_assets
from image_encoder import ImageEncoder
from runtime import load_images


def statistics(actual, reference, atol):
    assert actual.shape == reference.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    # Use full stored FP32 values, then evaluate the displayed criterion in FP64.
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    allowed = atol + 2e-4 * np.abs(reference.astype(np.float64))
    passed = bool(np.all(error <= allowed))
    assert passed
    assert np.allclose(actual, reference, atol=atol, rtol=2e-4)
    return {'shape': list(actual.shape), 'atol': atol, 'rtol': 2e-4,
            'max_abs_error': float(error.max()), 'mean_abs_error': float(error.mean()),
            'max_error_to_tolerance_ratio': float((error / allowed).max()),
            'elements_compared': int(error.size), 'elements_outside_tolerance': 0,
            'passed': passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/reference_comparison.json'))
    args = parser.parse_args()
    encoder = ImageEncoder(verify_assets())
    images = load_images()
    stages = {}
    with np.load(ROOT / 'assets/reference_outputs.npz', allow_pickle=False) as ref:
        calculations = [
            ('preprocess', lambda: images, 'preprocessed', None, 2e-6),
            ('patches', lambda: encoder.patchify(images[:2]), 'patches', 2, 2e-4),
            ('tokens', lambda: encoder.embed(images[:2]), 'input_tokens', 2, 2e-4),
            ('block0', lambda: encoder.blocks[0](encoder.pre_norm(encoder.embed(images[:2]))),
             'block0_tokens', 2, 2e-4),
            ('last_tokens', lambda: encoder.encode_tokens(images[:2]), 'last_tokens', 2, 2e-4),
            ('output', lambda: encoder(images), 'output_tokens', None, 2e-5),
            ('batch', lambda: encoder(images[[1, 0, 1]]), 'output_tokens', [1, 0, 1], 2e-5),
        ]
        output = None
        for name, calculate, key, selection, atol in calculations:
            expected = ref[key]
            if isinstance(selection, int):
                expected = expected[:selection]
            elif selection is not None:
                expected = expected[selection]
            actual = calculate()
            stages[name] = statistics(actual, expected, atol)
            if name == 'output':
                output = actual
            print(f'{name}: max(error/tolerance)='
                  f'{stages[name]["max_error_to_tolerance_ratio"]:.6f} '
                  f'outside=0/{stages[name]["elements_compared"]}', flush=True)

        reference = ref['output_tokens']
        errors = np.abs(output.astype(np.float64) - reference.astype(np.float64))
        worst = tuple(int(i) for i in np.unravel_index(errors.argmax(), errors.shape))
        near_zero = tuple(int(i) for i in np.unravel_index(np.abs(reference).argmin(), reference.shape))
        selected = [('first_value', (0, 0)), ('largest_absolute_error', worst),
                    ('reference_closest_to_zero', near_zero),
                    ('last_value', (len(output) - 1, output.shape[-1] - 1))]
        samples = []
        for reason, (row, feature) in selected:
            actual_value = float(output[row, feature])
            reference_value = float(reference[row, feature])
            error = abs(actual_value - reference_value)
            allowed = 2e-5 + 2e-4 * abs(reference_value)
            assert error <= allowed
            samples.append({'selection': reason, 'image_id': str(ref['ids'][row]),
                            'image_index_zero_based': row, 'feature_index_zero_based': feature,
                            'actual': actual_value, 'reference': reference_value,
                            'absolute_error': error, 'allowed_error': allowed, 'passed': True})

    result = {'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
              'criterion': 'abs(actual-reference) <= atol + rtol*abs(reference)',
              'assets_integrity_passed': True, 'stages': stages, 'output_samples': samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(samples, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

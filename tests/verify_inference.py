"""Check unseen inputs and a standalone CLI with weights but no teacher assets.

Run from the repository root:
    python tests/verify_inference.py --output results/extra_validation.json
The regular unit tests do not need course resources; this script needs weights.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from image_encoder import ImageEncoder
from preprocess_image import PreprocessImage
from runtime import load_weights


def validate_output(output, batch):
    assert output.shape == (batch, 512), output.shape
    assert output.dtype == np.float32, output.dtype
    assert np.isfinite(output).all()
    norms = np.sqrt((output * output).sum(-1))
    np.testing.assert_allclose(norms, 1.0, rtol=2e-4, atol=2e-4)
    return {'shape': list(output.shape), 'dtype': str(output.dtype),
            'min_l2_norm': float(norms.min()), 'max_l2_norm': float(norms.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/extra_validation.json'))
    args = parser.parse_args()
    encoder = ImageEncoder(load_weights())

    # These arrays are new inputs, independent of the ten teacher reference images.
    images = np.random.default_rng(137).uniform(-1, 1, (2, 3, 224, 224)).astype(np.float32)
    single_outputs = np.concatenate([encoder(images[i:i + 1]) for i in range(2)])
    order = [1, 0, 1, 0]
    batched_output = encoder(images[order])
    np.testing.assert_allclose(batched_output, single_outputs[order], rtol=2e-4, atol=2e-5)
    batch_summary = validate_output(batched_output, len(order))
    batch_summary.update({'input_seed': 137, 'order': order,
                          'max_abs_single_vs_batch': float(np.abs(
                              batched_output - single_outputs[order]).max())})

    # The standalone project contains no assets directory or reference outputs.
    # Place its temporary directory on the same volume so weights can be hard-linked.
    temporary_root = ROOT.parent / 'tmp'
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inference-check-', dir=temporary_root) as directory:
        work = Path(directory).resolve()
        assert work.is_relative_to(temporary_root.resolve())
        standalone = work / 'standalone'
        standalone.mkdir()
        for name in ('image_encoder.py', 'helpers.py', 'preprocess_image.py',
                     'runtime.py', 'run.py', 'config.json'):
            shutil.copyfile(ROOT / name, standalone / name)
        (standalone / 'weights').mkdir()
        source_weights = ROOT / 'weights' / 'image_encoder.npz'
        target_weights = standalone / 'weights' / 'image_encoder.npz'
        try:
            os.link(source_weights, target_weights)
        except OSError:
            shutil.copyfile(source_weights, target_weights)
        assert not (standalone / 'assets').exists()

        # Different dimensions and color modes exercise the supplied preprocessing.
        rgb_path, gray_path = work / 'rgb_pattern.png', work / 'grayscale_pattern.png'
        rows, columns = np.indices((180, 300))
        rgb = np.stack((columns % 256, rows % 256, (rows + columns) % 256), -1).astype(np.uint8)
        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(((rows[:120, :240] * 3 + columns[:120, :240]) % 256).astype(np.uint8)).save(gray_path)
        output_path = work / 'custom_tokens.npz'
        command = [sys.executable, '-X', 'utf8', '-B', str(standalone / 'run.py'), '--images',
                   str(rgb_path), str(gray_path), '--output', str(output_path)]
        process = subprocess.run(command, cwd=work, capture_output=True, text=True,
                                 encoding='utf-8', timeout=120)
        if process.returncode:
            raise RuntimeError(process.stdout + process.stderr)
        with np.load(output_path, allow_pickle=False) as data:
            ids = data['ids'].tolist()
            cli_output = data['output_tokens'].copy()
        assert ids == ['rgb_pattern', 'grayscale_pattern'], ids
        prepared = np.concatenate([PreprocessImage()(path) for path in (rgb_path, gray_path)])
        api_output = encoder(prepared)
        np.testing.assert_allclose(cli_output, api_output, rtol=2e-4, atol=2e-5)
        cli_summary = validate_output(cli_output, 2)
        cli_summary.update({'ids': ids, 'input_modes': ['RGB', 'L'],
                            'input_sizes': [[300, 180], [240, 120]],
                            'assets_present_in_standalone': False,
                            'working_directory_is_project_root': False,
                            'max_abs_cli_vs_api': float(np.abs(cli_output - api_output).max())})

    summary = {'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
               'single_vs_batch': batch_summary, 'standalone_cli': cli_summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

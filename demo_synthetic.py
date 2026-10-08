"""Generate a local-only registration demo, without any experimental images."""
import csv
import datetime as dt
import os
from pathlib import Path

import cv2
import numpy as np

from registration_pipeline import register_arrays, save_registration_result


def main():
    rng = np.random.default_rng(7)
    before = rng.normal(110, 24, (384, 384)).clip(0, 255).astype(np.uint8)
    for _ in range(80):
        x, y = rng.integers(15, 369, size=2)
        cv2.circle(before, (int(x), int(y)), int(rng.integers(2, 10)),
                   int(rng.integers(20, 235)), -1)
    after = cv2.warpAffine(before, np.float32([[1, 0, 9], [0, 1, -6]]),
                           (384, 384), borderMode=cv2.BORDER_REFLECT)
    result = register_arrays(before, after, 500, 500, use_ecc=True)
    workspace = Path(os.environ.get('SEM_WORKSPACE', Path(__file__).resolve().parent / 'workspace')).expanduser().resolve()
    folder = workspace / ('demo_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    files = save_registration_result(result, folder)
    row = {key: Path(files[key]).name for key in ('before', 'after', 'mask', 'diff')}
    row.update(pair_id='synthetic_translation', status='ok', annotation_dir='.',
               registration_metrics=Path(files['metrics']).name)
    index = folder / 'annotation_index.csv'
    with index.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print('Synthetic example only; no experimental images were used.')
    print(f'Inliers: {result.metrics.inliers}; NCC: {result.metrics.ncc_after:.4f}')
    print(f'Results: {folder}')
    print(f'Open with: python mask_editor_app.py --index "{index}"')


if __name__ == '__main__':
    main()

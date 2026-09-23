"""Download/cache official MNIST through torchvision; never loads pretrained models."""
from pathlib import Path
import argparse
import hashlib
import json
from torchvision.datasets import MNIST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    args = parser.parse_args()
    root = Path(args.data_path).resolve()
    datasets = [MNIST(str(root), train=train, download=True) for train in (True, False)]
    raw = root / 'MNIST/raw'
    records = []
    for path in sorted(raw.iterdir()):
        if path.is_file():
            records.append({'file': path.name, 'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    print(json.dumps({'train_examples': len(datasets[0]), 'test_examples': len(datasets[1]),
                      'files': records}, indent=2))


if __name__ == '__main__':
    main()

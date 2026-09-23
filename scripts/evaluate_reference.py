"""Reevaluate the distributed final MNIST checkpoint without relying on archived paths."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import torch
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mga_data import invert_trigger
from plato.models.lenet5 import Model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    expected = json.loads((ROOT / 'evidence/mnist_seed23/validated_result.json').read_text())['metrics']['after_recovery']
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha != expected['checkpoint_sha256']:
        parser.error('This is not the distributed reference checkpoint. Use validate_mga_run.py for a new run.')
    dataset = MNIST(str(Path(args.data_path).resolve()), train=False, download=False)
    x = (dataset.data.float().unsqueeze(1) / 255. - .1307) / .3081
    y = dataset.targets
    triggered = invert_trigger(x[y != 1], 'mnist')
    torch.set_num_threads(2)
    model = Model(num_classes=10)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(args.device).eval()
    correct = sum(int((model(x[i:i+256].to(args.device)).argmax(1).cpu() == y[i:i+256]).sum())
                  for i in range(0, len(y), 256))
    hits = sum(int((model(triggered[i:i+256].to(args.device)).argmax(1) == 1).sum())
               for i in range(0, len(triggered), 256))
    result = {'clean_correct': correct, 'clean_total': len(y), 'clean_accuracy': correct / len(y),
              'asr_hits': hits, 'asr_total': len(triggered), 'asr': hits / len(triggered),
              'checkpoint_sha256': sha,
              'matches_recorded_counts': correct == expected['clean_correct'] and hits == expected['asr_hits']}
    print(json.dumps(result, indent=2))
    if not result['matches_recorded_counts']:
        raise SystemExit('Inference counts differ: investigate environment/preprocessing; do not overwrite reference evidence.')


if __name__ == '__main__':
    main()

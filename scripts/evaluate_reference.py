"""Evaluate a supplied MNIST checkpoint; optional expected metrics are external."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import torch
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mga.data import invert_trigger
from plato.models.lenet5 import Model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--target-label', type=int, default=1)
    parser.add_argument('--expected-result', help='Optional validate_mga_run.py JSON for strict replay checks.')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not 0 <= args.target_label < 10:
        parser.error('--target-label must be in [0, 9].')
    expected = None
    if args.expected_result:
        expected = json.loads(Path(args.expected_result).read_text(encoding='utf8'))['metrics']['after_recovery']
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected is not None and sha != expected['checkpoint_sha256']:
        parser.error('Checkpoint hash differs from the supplied expected result.')
    dataset = MNIST(str(Path(args.data_path).resolve()), train=False, download=False)
    x = (dataset.data.float().unsqueeze(1) / 255. - .1307) / .3081
    y = dataset.targets
    triggered = invert_trigger(x[y != args.target_label], 'mnist')
    torch.set_num_threads(2)
    model = Model(num_classes=10)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(args.device).eval()
    correct = sum(int((model(x[i:i+256].to(args.device)).argmax(1).cpu() == y[i:i+256]).sum())
                  for i in range(0, len(y), 256))
    hits = sum(int((model(triggered[i:i+256].to(args.device)).argmax(1) == args.target_label).sum())
               for i in range(0, len(triggered), 256))
    result = {'clean_correct': correct, 'clean_total': len(y), 'clean_accuracy': correct / len(y),
              'asr_hits': hits, 'asr_total': len(triggered), 'asr': hits / len(triggered),
              'checkpoint_sha256': sha, 'target_label': args.target_label}
    if expected is not None:
        result['matches_recorded_counts'] = correct == expected['clean_correct'] and hits == expected['asr_hits']
    print(json.dumps(result, indent=2))
    if expected is not None and not result['matches_recorded_counts']:
        raise SystemExit('Inference counts differ: investigate environment/preprocessing; do not overwrite reference evidence.')


if __name__ == '__main__':
    main()

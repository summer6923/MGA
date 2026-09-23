"""Device fallback regressions for the packaged release; no data or GPU needed."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_config_under_test', ROOT / 'plato/config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Config = module.Config


class DeviceFallbackTests(unittest.TestCase):
    def setUp(self):
        self.original_instance = Config._instance
        self.original_args = getattr(Config, 'args', None)
        Config._instance = NS(trainer=NS())
        Config.args = NS(cpu=False, mps=False)
    def tearDown(self):
        Config._instance = self.original_instance
        if self.original_args is None:
            delattr(Config, 'args')
        else:
            Config.args = self.original_args
    def test_cpu_build_defaults_to_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(Config.device(), 'cpu')
    def test_available_gpu_preserves_cuda_zero(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
            self.assertEqual(Config.device(), 'cuda:0')
    def test_explicit_cpu_wins_over_available_gpu(self):
        Config.args.cpu = True
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
            self.assertEqual(Config.device(), 'cpu')


if __name__ == '__main__':
    unittest.main(verbosity=2)

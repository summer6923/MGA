"""Bounded publication-tree checks. Reports locations, never matched credential values.
This pattern scan is not a guarantee that every possible secret is detected.
"""
from pathlib import Path
import argparse
import ast
import json
import os
import re

SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'data', 'runs', 'outputs', '.pytest_cache'}
RULES = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'authenticated_url': re.compile(r'https?://[^\s/@:]+:[^\s/@]+@'),
    'literal_secret_assignment': re.compile(r'''(?i)\b(?:password|passwd|api_key|access_token|secret_key|aws_secret_access_key)\s*[:=]\s*["'][^"'\s]{6,}["']'''),
}
REQUIRED = ['LICENSE', 'NOTICE', 'README.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md',
            'MODIFICATIONS.md', 'requirements.txt', 'VERSION', 'configs/mga_mnist_full.yml',
            'configs/mga_cifar10_full.yml',
            'configs/mga_purchase_full.yml',
            'mga/data.py', 'mga/protocol.py', 'mga/reverse.py', 'mga/entry.py',
            'mga/evaluation.py', 'docs/implementation.md']


def inspect_text(text):
    return [{'rule': name, 'line': text.count('\n', 0, m.start()) + 1}
            for name, rule in RULES.items() for m in rule.finditer(text)]


def self_test():
    # Deliberately constructed sentinels, not stored credentials.
    cases = [('private_key', '-----BEGIN ' + 'PRIVATE KEY-----'),
             ('github_token', 'gh' + 'p_' + 'A' * 36),
             ('aws_access_key', 'AK' + 'IA' + 'A' * 16),
             ('authenticated_url', 'https://' + 'dummy:dummy@example.invalid'),
             ('literal_secret_assignment', 'pass' + 'word = "' + 'synthetic-value' + '"')]
    for name, sample in cases:
        if not any(x['rule'] == name for x in inspect_text(sample)):
            raise RuntimeError('Scanner self-test failed: ' + name)
    if inspect_text('password = os.getenv("PASSWORD")'):
        raise RuntimeError('Scanner misclassified dynamic environment lookup.')
    return len(cases) + 1


def check(root):
    findings, errors, files = [], [], []
    for name in REQUIRED:
        if not (root / name).is_file():
            errors.append('Missing required file: ' + name)
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                errors.append('Symlink not allowed in prepared release: ' + rel)
                continue
            files.append(path)
            low = name.lower()
            if (low == '.env' or low.startswith('.env.') or low.startswith(('id_rsa', 'id_ed25519'))
                    or path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}):
                findings.append({'path': rel, 'rule': 'sensitive_filename', 'line': 0})
            if path.stat().st_size > 10 * 1024 * 1024:
                errors.append('Unexpected oversized source asset: ' + rel)
                continue
            try:
                text = path.read_text(encoding='utf8')
            except UnicodeError:
                errors.append('Unexpected binary in source-only package: ' + rel)
                continue
            findings.extend(dict(path=rel, **hit) for hit in inspect_text(text))
            if path.suffix == '.py':
                try:
                    ast.parse(text, filename=rel)
                except SyntaxError as error:
                    errors.append(f'Python syntax error in {rel}:{error.lineno}')
            if path.suffix == '.md':
                for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                    if '://' in target or target.startswith(('mailto:', '#')):
                        continue
                    local = target.split('#', 1)[0]
                    if local and not (path.parent / local).exists():
                        errors.append(f'Broken local documentation link in {rel}: {local}')
    return {'passed': not findings and not errors, 'source_files_scanned': len(files),
            'credential_candidate_count': len(findings),
            'rules': sorted(RULES), 'findings': findings, 'errors': errors,
            'scope': 'Prepared source tree only; ignored local environments/data/runs and no Git history.',
            'limitation': 'No supported credential patterns found is not proof of absence of every possible secret.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    result = check(Path(args.root).resolve())
    if args.self_test:
        result['scanner_self_tests_passed'] = self_test()
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

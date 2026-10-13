#!/usr/bin/env python3
"""Restore the tested local dependency changes without overwriting other edits."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate without changing sources')
    parser.add_argument('--only', choices=('tfhe-rs', 'heongpu', 'rngongpu'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    versions = json.loads((root / 'versions.json').read_text())
    names = [args.only] if args.only else ['heongpu', 'rngongpu', 'tfhe-rs']
    for name in names:
        spec = versions[name]
        repo = root / spec['path']
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        if head != spec['commit']:
            raise RuntimeError(f'{name}: expected {spec["commit"]}, found {head}')
        patch = root / 'patches' / f'{name}.patch'
        command = ['git', 'apply', '--check', str(patch)]
        check = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        if check.returncode == 0:
            if not args.check:
                subprocess.run(['git', 'apply', str(patch)], cwd=repo, check=True)
            print(f'{name}: {"applicable" if args.check else "applied"}')
            continue
        reverse = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)],
                                 cwd=repo, capture_output=True, text=True)
        if reverse.returncode:
            raise RuntimeError(f'{name}: patch conflicts with local edits; no files replaced.\n{check.stderr}')
        print(f'{name}: already applied')


if __name__ == '__main__':
    main()

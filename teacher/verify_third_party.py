import hashlib
import json
import os
import sys

from .official import THIRD_PARTY

def main():
    with open(os.path.join(THIRD_PARTY, 'MANIFEST.json')) as f:
        manifest = json.load(f)
    bad = 0
    for rel, digest in sorted(manifest['files'].items()):
        path = os.path.join(THIRD_PARTY, rel)
        if not os.path.isfile(path):
            print(f'MISSING  {rel}')
            bad += 1
            continue
        with open(path, 'rb') as f:
            ok = hashlib.sha256(f.read()).hexdigest() == digest
        print(f"{'ok      ' if ok else 'MODIFIED'} {rel}")
        bad += 0 if ok else 1
    for name, repo in manifest['repos'].items():
        print(f"{name}: {repo['url']} @ {repo['commit']} ({repo['license']})")
    print('all vendored files match upstream' if bad == 0 else f'{bad} file(s) differ from upstream')
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()

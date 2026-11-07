import importlib
import sys
import time
import traceback

MODULES = ['test_networks', 'test_losses', 'test_data', 'test_phema', 'test_pilot', 'test_pipeline']

def main(argv=None):
    names = (argv if argv is not None else sys.argv[1:]) or MODULES
    failures, passed = [], 0
    for m in names:
        mod_name, _, only = m.partition('::')
        mod = importlib.import_module(f'teacher.tests.{mod_name}')
        tests = [n for n in dir(mod) if n.startswith('test_') and callable(getattr(mod, n))]
        tests.sort(key=lambda n: getattr(mod, n).__code__.co_firstlineno)
        for name in tests:
            if only and name != only:
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                passed += 1
                print(f'PASS {mod_name}.{name} ({time.time() - t0:.1f} s)', flush=True)
            except Exception:
                traceback.print_exc()
                failures.append(f'{mod_name}.{name}')
                print(f'FAIL {mod_name}.{name} ({time.time() - t0:.1f} s)', flush=True)
    print(f'\n{passed} passed, {len(failures)} failed' + (': ' + ', '.join(failures) if failures else ''))
    return 1 if failures else 0

if __name__ == '__main__':
    sys.exit(main())

"""Resumable analysis stages and reversible archival of superseded results."""
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common.paths import PROJECT_DIR


def archive_paths(paths, *, dry_run=False):
    paths = [Path(p) for p in paths if Path(p).exists()]
    # Avoid moving a child twice if its parent is already selected.
    paths = sorted(set(paths), key=lambda p: len(p.parts))
    selected = []
    for path in paths:
        if not any(path == parent or parent in path.parents for parent in selected):
            selected.append(path)
    if not selected:
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = PROJECT_DIR / 'archive/reruns' / stamp
    for path in selected:
        path = path.absolute()
        if path.is_symlink():
            raise ValueError(f'Refusing to archive a symlink as an analysis directory: {path}')
        try:
            relative = path.relative_to(PROJECT_DIR)
        except ValueError:
            # External custom RUN directories stay recoverable without flattening names.
            relative = Path('external') / hashlib.sha256(str(path).encode()).hexdigest()[:12] / path.name
        target = destination / relative
        print(f'Archive {path} -> {target}', flush=True)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            log = PROJECT_DIR / 'logs/reruns.tsv'
            log.parent.mkdir(exist_ok=True)
            with log.open('a') as stream:
                stream.write(f'{path}\t{target}\n')


def command(cmd, *, dry_run=False):
    print('$ ' + ' '.join(map(str, cmd)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, cmd)), cwd=PROJECT_DIR, check=True)


def recipe_signature(cmd):
    """Fingerprint code that generates statistics, excluding plot-only styling."""
    digest = hashlib.sha256(json.dumps(list(map(str, cmd))).encode())
    scientific = ['config.py', 'geo.py', 'environment.py', 'matching.py', 'stats.py', 'fe_logit.py']
    files = [PROJECT_DIR / 'common' / name for name in scientific]
    files += [Path(arg) for arg in cmd if str(arg).endswith('.py') and Path(arg).is_file()]
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stage_complete(outputs):
    return all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in outputs)


def run_stage(name, cmd, outputs, state, *, dry_run=False, rerun=False):
    signature = recipe_signature(cmd)
    prior = state.get(name)
    if prior and prior['recipe'] != signature:
        raise ValueError(f'Stage {name} was registered with different code/settings. Use --force to archive and rerun.')
    if not rerun and stage_complete(outputs):
        print(f'[{name}] complete; reusing saved results', flush=True)
        if not dry_run and prior is None:
            state[name] = {'recipe': signature, 'origin': 'adopted_existing', 'generated_with': None}
        return False
    command(cmd, dry_run=dry_run)
    if not dry_run:
        if not stage_complete(outputs):
            raise RuntimeError(f'Stage {name} finished without all expected outputs')
        state[name] = {'recipe': signature, 'origin': 'generated',
                       'completed_utc': datetime.now(timezone.utc).isoformat()}
    return True

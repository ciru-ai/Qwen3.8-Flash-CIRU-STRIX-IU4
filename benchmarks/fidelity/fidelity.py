#!/usr/bin/env python3
"""Frozen engineering-text fidelity; no model outputs are executed."""
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
VOCAB = 248320
WINDOWS = 16
POSITIONS = 1024


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_data():
    for name, digest in read(DATA / 'files.json').items():
        require(sha(DATA / name) == digest, f'Dataset hash mismatch: {name}')
    return read(DATA / 'panel.json')


def fetch_reference(cache):
    verify_data()
    manifest = read(DATA / 'reference.json')
    cache.mkdir(parents=True, exist_ok=True)
    for item in manifest['files']:
        path = cache / item['file']
        if path.exists():
            require(sha(path) == item['sha256'], f'Corrupt cached reference: {path}; remove it and fetch again')
            continue
        print('Downloading', item['file'], flush=True)
        temp = path.with_suffix('.part')
        try:
            with urllib.request.urlopen(manifest['release_base'] + '/' + item['file'], timeout=120) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            require(sha(temp) == item['sha256'], 'Downloaded reference checksum failed')
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def log_probs(logits):
    x = np.asarray(logits, dtype=np.float64)
    require(np.isfinite(x).all(), 'Nonfinite logits')
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def compare(reference, candidate, labels):
    r = np.asarray(reference, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    rlp, clp = log_probs(r), log_probs(c)
    rows = np.arange(len(labels))
    ctop = c.argmax(axis=1)
    kl = (np.exp(rlp) * (rlp - clp)).sum(axis=1)
    require(float(kl.min()) > -1e-8, 'Invalid negative KL')
    return {'kl': kl, 'top1': (ctop == r.argmax(axis=1)).astype(float),
            'tie_aware': (r[rows, ctop] == r.max(axis=1)).astype(float),
            'nll': -clp[rows, labels]}


def summarize(metrics):
    draws = np.random.default_rng(20260916).integers(0, 8, size=(10000, 8))
    def summary(x, scale=1):
        clusters = x.reshape(8, 2048).mean(axis=1)
        return {'mean': float(x.mean() * scale), 'ci95': np.percentile(clusters[draws].mean(axis=1) * scale, [2.5, 97.5]).tolist()}
    result = {out: summary(metrics[key], scale) for out, key, scale in [
        ('mean_kl_nats', 'kl', 1), ('top1_agreement_percent', 'top1', 100),
        ('tie_aware_agreement_percent', 'tie_aware', 100), ('observed_nll', 'nll', 1)]}
    result['perplexity'] = {'value': math.exp(result['observed_nll']['mean']),
                            'ci95': [math.exp(x) for x in result['observed_nll']['ci95']]}
    result['top1_matches'] = int(metrics['top1'].sum())
    result['tie_aware_matches'] = int(metrics['tie_aware'].sum())
    return result


def score(capture, cache, label, package_bytes):
    panel = verify_data()
    reference = read(DATA / 'reference.json')
    require(reference['panel_sha256'] == panel['panel_sha256'], 'Reference/panel mismatch')
    require(package_bytes > 0, 'Package size must be positive')
    manifest = read(capture / 'manifest.json')
    require(manifest['status'] == 'PASS' and manifest['output_shape'] == [16, 1024, VOCAB], 'Incomplete capture')
    logits = capture / 'logits.f32le'
    require(logits.stat().st_size == 16 * 1024 * VOCAB * 4, 'Logit file size mismatch')
    for w in range(16):
        actual = (capture / f'window-{w}.txt').read_bytes().decode('utf-8', errors='replace')
        require(actual == (DATA / f'window-{w:02d}.txt').read_text(), f'Tokenizer mismatch in window {w}')
    ids = np.fromfile(DATA / 'panel.tokens.i32le', dtype='<i4').reshape(16, 2049)
    candidate = np.memmap(logits, mode='r', dtype='<f4', shape=(16, 1024, VOCAB))
    metrics = {k: np.empty(16384, dtype=np.float64) for k in ['kl', 'top1', 'tie_aware', 'nll']}
    windows = []
    for item in reference['files']:
        w = item['window']; path = cache / item['file']
        require(sha(path) == item['sha256'], f'Reference hash failed: {path}')
        with gzip.open(path, 'rb') as f:
            raw = f.read(item['uncompressed_bytes'] + 1)
        require(len(raw) == item['uncompressed_bytes'] and hashlib.sha256(raw).hexdigest() == item['uncompressed_sha256'], 'Reference payload failed')
        teacher = np.frombuffer(raw, dtype='<u2').astype('<u4')
        teacher <<= 16
        teacher = teacher.view('<f4').reshape(1024, VOCAB)
        for start in range(0, 1024, 32):
            end = start + 32
            m = compare(teacher[start:end], candidate[w, start:end], ids[w, 1025+start:1025+end])
            for key in metrics:
                metrics[key][w*1024+start:w*1024+end] = m[key]
        windows.append({'window': w, 'repository': panel['windows'][w]['repository'],
                        **{k: float(v[w*1024:(w+1)*1024].mean()) for k,v in metrics.items()}})
        print(f'Scored window {w+1}/16', flush=True)
        del teacher, raw
    result = {'status': 'PASS', 'label': label, 'size_bytes': package_bytes, 'metrics': summarize(metrics),
              'panel_sha256': panel['panel_sha256'], 'reference_original_f32_sha256': reference['original_f32_sha256'],
              'candidate_logits_sha256': sha(logits), 'positions': 16384, 'classification': 'local-custom',
              'per_window': windows, 'capture': manifest,
              'limitations': ['Engineering text, not task accuracy.', 'Eight independent repository clusters.',
                              'Full configuration fidelity, not isolated weight error.', 'Not comparable to the publisher chart on another corpus/reference.']}
    save(capture.parent / 'result.json', result)
    return result


def build(source, runtime, output):
    source, runtime = source.resolve(), runtime.resolve()
    command = [os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', str(ROOT / 'capture_logits.cpp'),
               '-I'+str(source/'include'), '-I'+str(source/'ggml/include'), '-I'+str(source/'vendor'),
               '-L'+str(runtime), '-Wl,-rpath,'+str(runtime), '-lllama', '-lggml', '-lggml-base', '-o', str(output)]
    subprocess.run(command, check=True)


def run(args):
    verify_data()
    require(not args.out.exists(), 'Output exists; use a new directory. First attempts are never overwritten.')
    require(args.model.is_file(), 'Model does not exist')
    require(shutil.disk_usage(args.out.parent).free > 17 * 2**30, 'Need at least 17 GiB free for candidate logits')
    require(Path('/dev/kfd').exists(), 'AMD /dev/kfd is unavailable; this runner targets Linux HIP/ROCm')
    ref = read(DATA / 'reference.json')
    require(all((args.cache / x['file']).exists() for x in ref['files']), 'Run the fetch command before capture')
    args.out.mkdir()
    env = {k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_', 'HSA_', 'DEBUG_HIP', 'ENABLE_RETAINED'))}
    env['GGML_QWEN4EXP_PLE_IO_WORKERS'] = '32'
    env['HIP_VISIBLE_DEVICES'] = str(args.device)
    env['LD_LIBRARY_PATH'] = str(args.runtime.resolve()) + ':' + env.get('LD_LIBRARY_PATH', '')
    model_files = [args.model] + args.extra_model_file
    if args.ple:
        require(args.ple.is_dir(), 'PLE must be the sidecar directory')
        model_files += sorted(p for p in args.ple.rglob('*') if p.is_file())
    model_files = list(dict.fromkeys(p.resolve() for p in model_files))
    print('Hashing model package...', flush=True)
    identity = [{'file': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in model_files]
    exe = args.out / 'capture_logits'
    build(args.source, args.runtime, exe)
    command = [str(exe.resolve()), str(args.runtime.resolve()), str(args.model.resolve()),
               str(args.ple.resolve()) if args.ple else 'builtin', str(DATA/'panel.tokens.i32le'),
               str((args.out/'capture').resolve()), args.label, '16']
    receipt = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host': platform.platform(),
               'model_files': identity, 'command': command, 'capture_source_sha256': sha(ROOT/'capture_logits.cpp'),
               'runtime_libraries': {p.name: sha(p) for p in args.runtime.glob('*.so*') if p.is_file() and not p.is_symlink()},
               'environment': {k:v for k,v in env.items() if k.startswith(('LLAMA_', 'GGML_', 'HIP_', 'HSA_'))}}
    receipt['capture_executable_sha256'] = sha(exe)
    for key, command_git in [('runtime_source_commit', ['git','-C',str(args.source),'rev-parse','HEAD']), ('runtime_source_changes', ['git','-C',str(args.source),'status','--porcelain'])]:
        got = subprocess.run(command_git, text=True, capture_output=True)
        receipt[key] = got.stdout.strip() if got.returncode == 0 else 'unavailable (source archive)'
    receipt['memory_total'] = next((line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')), 'unknown')
    save(args.out/'run.json', receipt)
    with (args.out/'capture.log').open('x') as log:
        status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    require(status == 0, f'Capture failed ({status}); evidence retained in {args.out}/capture.log')
    score(args.out/'capture', args.cache, args.label, sum(p['bytes'] for p in identity))
    subprocess.run([sys.executable, str(ROOT/'plot.py'), '--result', str(args.out/'result.json'), '--out', str(args.out/'comparison')], check=True)
    print('Complete:', args.out/'result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fetch', help='Download checksum-verified, lossless BF16 reference shards')
    p.add_argument('--cache', type=Path, default=ROOT/'reference-cache')
    p = sub.add_parser('run', help='Capture all16 windows, score, and plot')
    for name in ['model', 'runtime', 'source', 'out']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--label', required=True);p.add_argument('--ple', type=Path)
    p.add_argument('--extra-model-file', type=Path, action='append', default=[], help='Additional GGUF shards for package hash and size')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--cache', type=Path, default=ROOT/'reference-cache')
    p = sub.add_parser('score', help='Score a complete capture without running GPU inference')
    p.add_argument('--capture', type=Path, required=True);p.add_argument('--cache', type=Path, default=ROOT/'reference-cache')
    p.add_argument('--label', required=True);p.add_argument('--package-bytes', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'fetch': fetch_reference(args.cache)
    elif args.command == 'run': run(args)
    else: score(args.capture, args.cache, args.label, args.package_bytes)


if __name__ == '__main__':
    try: main()
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f'ERROR: {e}', file=sys.stderr);sys.exit(1)

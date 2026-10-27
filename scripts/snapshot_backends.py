"""Freeze tracked backend runtime files from explicit local Git checkouts."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--ddsp', required=True, type=Path)
ap.add_argument('--anime', required=True, type=Path)
ap.add_argument('--rvc', required=True, type=Path)
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = {'format': 1, 'note': 'Pinned runtime snapshots; no weights, media, environments or local configuration.', 'sources': []}
existing = root/'vendor/manifest.json'
if existing.is_file():
    manifest['externalEngines'] = json.loads(existing.read_text(encoding='utf-8')).get('externalEngines', [])
for name, src, url in [
    ('ddsp-pipeline', args.ddsp, 'https://github.com/InariNGT-Ofiicial/dsh-train-pipeline'),
    ('anime-voice', args.anime, 'https://github.com/InariNGT-Ofiicial/dsh-anime-voice'),
    ('rvc-voice-kit', args.rvc, 'https://github.com/InariNGT-Ofiicial/rvc-voice-kit'),
]:
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(src), *argv])
    commit = git('rev-parse', 'HEAD').decode().strip()
    paths = git('ls-tree', '-r', '--name-only', 'HEAD').decode().splitlines()
    records = []
    for rel in paths:
        if name == 'ddsp-pipeline':
            keep = rel.startswith(('src/', 'py/', 'docs/', 'materials/')) or rel in ['package.json', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md']
        elif name == 'anime-voice':
            keep = rel.startswith(('skills/anime-voice-line-extractor/scripts/', 'skills/anime-voice-clone-training/scripts/', 'plugins/tts-train-pipeline/src/', 'plugins/tts-train-pipeline/py/')) or rel in ['README.md', 'requirements-analysis.txt', 'plugins/tts-train-pipeline/package.json', 'plugins/tts-train-pipeline/README.md']
        else:
            keep = rel in ['rvcx/__init__.py', 'rvcx/paths.py', 'rvcx/voice.py', 'scripts/train/p1_run.py', 'README.md', 'LICENSE']
        if not keep or rel.endswith(('.pyc', '.wav', '.mp3', '.pth', '.ckpt')):
            continue
        data = git('show', f'HEAD:{rel}')
        dst = root / 'vendor' / name / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        record = {'path': rel, 'sha256': hashlib.sha256(data).hexdigest()}
        if name == 'anime-voice' and rel == 'plugins/tts-train-pipeline/py/scan_corpus.py':
            text = data.decode('utf-8')
            needle = '            src = ""\n            if isinstance(v, str) and v:'
            replacement = '            src = str(r.get("voice_path") or "")\n            if not os.path.isfile(src):\n                src = ""\n            if not src and isinstance(v, str) and v:'
            normalized = text.replace('\r\n', '\n')
            if normalized.count(needle) != 1:
                raise RuntimeError('Upstream scan_corpus.py changed; review voice_path compatibility patch')
            data = normalized.replace(needle, replacement).encode('utf-8')
            record['patchedSha256'] = hashlib.sha256(data).hexdigest()
            record['patch'] = 'Prefer exact existing voice_path before basename lookup, matching build_tts_dataset.py and the anime bridge.'
        if name == 'rvc-voice-kit' and rel == 'scripts/train/p1_run.py':
            text = data.decode('utf-8').replace('\r\n', '\n')
            text = text.replace('    if small:\n', '    if small and not a.resume:\n', 1)
            needle = '    ap.add_argument("--pitch", type=int, default=0)'
            if text.count(needle) != 1:
                raise RuntimeError('RVC runner changed; review explicit resume patch')
            text = text.replace(needle, '    ap.add_argument("--resume", action="store_true", help="Explicitly resume an existing experiment")\n'+needle)
            data = text.encode('utf-8')
            record['patchedSha256'] = hashlib.sha256(data).hexdigest()
            record['patch'] = 'An explicit --resume bypasses the old exported-model shortcut; the unified adapter validates experiment identity and artifacts before/after execution.'
        dst.write_bytes(data)
        records.append(record)
    manifest['sources'].append({'id': name, 'repository': url, 'commit': commit, 'files': records})
(root / 'vendor/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps([{'source': s['id'], 'commit': s['commit'], 'files': len(s['files'])} for s in manifest['sources']]))

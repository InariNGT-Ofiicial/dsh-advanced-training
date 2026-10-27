"""Checked RVC training stages and owned background jobs; no inference dispatch."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from rvc_resume import backup_resume, guarded_runner

AUDIO = {'.wav', '.flac', '.ogg', '.opus', '.mp3', '.m4a', '.aif', '.aiff'}
KIT = Path(__file__).resolve().parents[1] / 'vendor/rvc-voice-kit'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).is_file() else default


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def context(request):
    c = request['config']
    p = request.get('params', {})
    exp = p.get('expName') or c.get('expName', 'my-rvc-voice')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', exp), 'expName 只能用英文、数字、短横线或下划线，不能是路径。角色显示名可用中文。')
    repo = Path(c['repo']).resolve()
    sr = p.get('sampleRate') or c.get('sampleRate', '48k')
    require(sr in ['48k', '40k'], '本路线只接入 v2 / 48k 或 40k。')
    work = Path(c['workRoot']).resolve() / exp
    return c, p, exp, repo, repo/'logs'/exp, work, sr


def source_audio(path):
    root = Path(path).resolve()
    require(root.is_dir(), f'单角色语音目录不存在：{root}')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO)
    require(files, f'目录里没有可用音频：{root}')
    return root, files


def inventory(path, fingerprint=False):
    import soundfile as sf
    root, files = source_audio(path)
    rows, errors = [], []
    for f in files:
        try:
            i = sf.info(f)
            require(i.frames > 0 and i.samplerate > 0, '空音频')
            row = {'file': str(f), 'seconds': i.frames/i.samplerate, 'sampleRate': i.samplerate, 'channels': i.channels}
            if fingerprint:
                row['sha256'] = sha(f)
            rows.append(row)
        except Exception as e:
            errors.append({'file': str(f), 'error': str(e)})
    digest = hashlib.sha256(json.dumps([(str(Path(r['file']).relative_to(root)), r.get('sha256')) for r in rows], ensure_ascii=False).encode()).hexdigest() if fingerprint else None
    return {'ok': bool(rows) and not errors, 'input': str(root), 'count': len(rows), 'seconds': sum(r['seconds'] for r in rows), 'errors': errors, 'files': rows, 'fingerprint': digest}


def data_proof(logdir, sr, min_slices=1):
    import soundfile as sf
    target = 48000 if sr == '48k' else 40000
    gt = sorted((logdir/'0_gt_wavs').glob('*.wav'))
    low = sorted((logdir/'1_16k_wavs').glob('*.wav'))
    require(len(gt) >= min_slices and len(low) == len(gt), f'预处理产物不足或不配对：gt={len(gt)}, 16k={len(low)}，至少 {min_slices}')
    require({p.stem for p in gt} == {p.stem for p in low}, '训练音频与 16k 音频文件名未配对。')
    for p in gt + low:
        i = sf.info(p)
        require(i.frames > 0 and i.channels == 1 and i.samplerate == (16000 if p.parent.name == '1_16k_wavs' else target), f'音频规格或内容错误：{p}')
    return {'slices': len(gt), 'sampleRate': target}


def feature_proof(logdir, repo, sr):
    import numpy as np
    data = data_proof(logdir, sr)
    for gt in (logdir/'0_gt_wavs').glob('*.wav'):
        for folder in ['3_feature768', '2a_f0', '2b-f0nsf']:
            base = logdir/folder
            p = next((x for x in [base/(gt.stem+'.npy'), base/(gt.stem+'.wav.npy')] if x.is_file()), None)
            require(p is not None, f'缺少 {folder}：{gt.stem}')
            a = np.load(p, allow_pickle=False)
            require(a.size > 0 and np.isfinite(a).all(), f'空或非有限特征：{p}')
            if folder == '3_feature768':
                require(a.ndim == 2 and a.shape[1] == 768, f'v2 特征维度不对：{p}')
    listing = logdir/'filelist.txt'
    require(listing.is_file(), '缺少 filelist.txt')
    rows = [l for l in listing.read_text(encoding='utf-8').splitlines() if l.strip()]
    require(len(rows) >= data['slices'] + 2, '训练清单不完整或缺少静音样本。')
    for line in rows:
        fields = line.split('|')
        require(len(fields) == 5 and fields[-1] == '0', '训练清单必须是单说话人、5 列。')
        for value in fields[:4]:
            require(Path(value).is_file() and Path(value).stat().st_size > 0, f'清单产物不存在：{value}')
    return {**data, 'listRows': len(rows), 'filelist': str(listing), 'featureDim': 768}


def model_candidates(repo, exp):
    def key(p):
        m = re.fullmatch(re.escape(exp)+r'_e(\d+)_s(\d+)\.pth', p.name)
        return (int(m[1]), int(m[2])) if m else (-1, -1)
    return sorted([p for p in (repo/'assets/weights').glob(exp+'_e*.pth') if key(p)[0] >= 0], key=key)


def inspect_model(path, sr):
    import torch
    d = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(d, dict) and isinstance(d.get('weight'), dict) and d['weight'], '不是 RVC 导出的小模型（不能用 G_/D_ 训练检查点代替）。')
    require(d.get('version') == 'v2' and d.get('f0') == 1 and d.get('sr') == sr, '模型 version/f0/sr 与实验不一致。')
    require(d.get('config') and d['config'][-1] == (48000 if sr == '48k' else 40000), '模型内部采样率不一致。')
    for name, value in d['weight'].items():
        require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f'模型含无效权重：{name}')
    return {'file': str(path), 'sha256': sha(path), 'version': 'v2', 'f0': 1, 'sampleRate': d['config'][-1], 'info': d.get('info'), 'bytes': Path(path).stat().st_size}


def index_proof(logdir):
    import faiss
    import numpy as np
    paths = sorted(logdir.glob('added_*.index'), key=lambda p: p.stat().st_mtime_ns)
    require(paths, '没有 added_*.index；trained_*.index 不能代替成品检索索引。')
    path = paths[-1]
    idx = faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8).copy())
    require(idx.d == 768 and idx.ntotal > 0, '索引为空或维度不是 v2 的 768。')
    return {'file': str(path), 'sha256': sha(path), 'dimensions': idx.d, 'vectors': idx.ntotal}


def doctor(request):
    c, p, exp, repo, logdir, work, sr = context(request)
    checks = []
    def check(name, ok, detail): checks.append({'name': name, 'ok': bool(ok), 'detail': str(detail)})
    for name in ['train/preprocess.py', 'train/dataset/extract_f0.py', 'train/dataset/extract_hubert_feature.py', 'train/train.py', 'train/train_index.py']:
        check(name, (repo/name).is_file(), repo/name)
    hf_hubert = (repo/'assets/hubert_base/config.json').is_file() and any((repo/'assets/hubert_base'/f).is_file() for f in ['pytorch_model.bin', 'model.safetensors'])
    check('HuBERT/ContentVec', hf_hubert or (repo/'assets/hubert/hubert_base.pt').is_file(), 'assets/hubert_base (Transformers) or assets/hubert/hubert_base.pt (legacy)')
    for name in [f'assets/pretrained_v2/f0G{sr}.pth', f'assets/pretrained_v2/f0D{sr}.pth', ('configs/v1/' if sr == '40k' else 'configs/v2/')+sr+'.json']:
        check(name, (repo/name).is_file(), repo/name)
    method = c.get('f0Method', 'rmvpe')
    if method == 'rmvpe': check('rmvpe', (repo/'assets/rmvpe/rmvpe.pt').is_file(), 'assets/rmvpe/rmvpe.pt')
    for name in ['torch', 'numpy', 'soundfile', 'faiss', 'librosa', 'parselmouth']:
        check(name, importlib.util.find_spec(name) is not None, name)
    for rel in [f'0_gt_wavs/mute{sr}.wav', '3_feature768/mute.npy', '2a_f0/mute.wav.npy', '2b-f0nsf/mute.wav.npy']:
        check('mute '+rel, (repo/'logs/mute'/rel).is_file(), rel)
    try:
        ff = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
        check('ffmpeg in child PATH', ff.returncode == 0, 'ffmpeg')
    except Exception as e: check('ffmpeg in child PATH', False, e)
    if p.get('deep'):
        import torch
        check('CUDA GPU 0', torch.cuda.is_available(), torch.__version__)
    return {'ok': all(x['ok'] for x in checks), 'stage': 'rvc-doctor', 'checks': checks, 'repo': str(repo), 'python': sys.executable, 'sampleRate': sr, 'version': 'v2'}


def register(request):
    c, p, exp, repo, logdir, work, sr = context(request)
    with experiment_lock(logdir, 'register-'+uuid.uuid4().hex):
        return register_unlocked(request)


def register_unlocked(request):
    c, p, exp, repo, logdir, work, sr = context(request)
    record = read_json(work/'experiment.json', {})
    require(record and record.get('repo') == str(repo) and record.get('sampleRate') == sr, '只能交接本入口建立、工程和采样率对应的实验。')
    models = model_candidates(repo, exp)
    require(models, '没有本实验导出的 RVC 小模型。')
    model = Path(p.get('model') or models[-1]).resolve()
    require(model in [m.resolve() for m in models], '只能登记本实验导出的小模型，不能混用其他实验。')
    detail = inspect_model(model, sr)
    idx = index_proof(logdir) if p.get('includeIndex', True) else None
    config = logdir/'config.json'
    require(config.is_file() and (logdir/'filelist.txt').is_file(), '模型交接缺少对应 config.json/filelist.txt。')
    require(read_json(config)['data']['sampling_rate'] == detail['sampleRate'], '训练配置与模型采样率不对应。')
    target = Path(p.get('outDir') or (Path(c['modelDir'])/exp)).resolve()
    require(not target.exists(), f'不覆盖已有音色包：{target}')
    temp = target.with_name(target.name+'.tmp-'+uuid.uuid4().hex)
    temp.mkdir(parents=True)
    files = [(model, 'model.pth'), (config, 'config.json'), (logdir/'filelist.txt', 'filelist.txt')]
    if idx: files.append((Path(idx['file']), Path(idx['file']).name))
    training_log = work/'stage-train.log'
    if not training_log.is_file(): training_log = logdir/'train.log'
    require(training_log.is_file(), '缺少本实验训练日志，不能把单个模型文件冒充完整交接包。')
    files.append((training_log, 'training.log'))
    artifacts = []
    for source, name in files:
        shutil.copy2(source, temp/name)
        artifacts.append({'file': name, 'sha256': sha(temp/name), 'bytes': (temp/name).stat().st_size})
    manifest = {'format': 1, 'route': 'rvc', 'artifact_kind': 'rvc_voice_model', 'role': p.get('role') or exp,
                'model': {**detail, 'file': 'model.pth', 'epochs': int(re.search(r'_e(\d+)_s', model.name)[1])},
                'index': ({**idx, 'file': Path(idx['file']).name} if idx else None),
                'training': {'experiment': exp, 'source': read_json(work/'experiment.json', {}), 'configSha256': sha(config)},
                'artifacts': artifacts, 'qualityAccepted': False}
    write_json(temp/'.rvc-manifest.json', manifest)
    os.rename(temp, target)
    return {'ok': True, 'stage': 'rvc-register', 'outDir': str(target), 'manifest': str(target/'.rvc-manifest.json'), 'artifacts': artifacts, 'qualityAccepted': False}


def is_alive(pid):
    if not pid: return False
    if os.name == 'nt':
        import ctypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.OpenProcess.restype = ctypes.c_void_p
        handle = k.OpenProcess(0x1000, False, int(pid))
        if not handle: return ctypes.get_last_error() == 5
        code = ctypes.c_ulong()
        try: return bool(k.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally: k.CloseHandle(ctypes.c_void_p(handle))
    try: os.kill(int(pid), 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


class OwnedProcessTree:
    """Windows job handle also ends descendants if the coordinator crashes."""
    def __enter__(self):
        self.handle = None
        if os.name == 'nt':
            import ctypes as ct
            from ctypes import wintypes as wt
            class Basic(ct.Structure):
                _fields_ = [('processTime', ct.c_int64), ('jobTime', ct.c_int64), ('flags', wt.DWORD),
                            ('minWorkingSet', ct.c_size_t), ('maxWorkingSet', ct.c_size_t), ('activeLimit', wt.DWORD),
                            ('affinity', ct.c_size_t), ('priority', wt.DWORD), ('scheduling', wt.DWORD)]
            class IO(ct.Structure):
                _fields_ = [(n, ct.c_uint64) for n in ['readOps', 'writeOps', 'otherOps', 'readBytes', 'writeBytes', 'otherBytes']]
            class Limits(ct.Structure):
                _fields_ = [('basic', Basic), ('io', IO), ('processMemory', ct.c_size_t), ('jobMemory', ct.c_size_t),
                            ('peakProcessMemory', ct.c_size_t), ('peakJobMemory', ct.c_size_t)]
            self.k = ct.WinDLL('kernel32', use_last_error=True)
            self.k.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
            self.k.CreateJobObjectW.restype = wt.HANDLE
            self.k.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
            self.k.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
            self.k.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
            self.k.CloseHandle.argtypes = [wt.HANDLE]
            self.handle = self.k.CreateJobObjectW(None, None)
            if not self.handle: raise ct.WinError(ct.get_last_error())
            limits = Limits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.k.SetInformationJobObject(self.handle, 9, ct.byref(limits), ct.sizeof(limits)):
                error = ct.get_last_error()
                self.k.CloseHandle(self.handle)
                self.handle = None
                raise ct.WinError(error)
        return self

    def attach(self, child):
        if self.handle and not self.k.AssignProcessToJobObject(self.handle, int(child._handle)):
            import ctypes
            error = ctypes.get_last_error()
            child.kill()
            child.wait(timeout=30)
            raise ctypes.WinError(error)

    def terminate(self, child):
        if self.handle:
            if not self.k.TerminateJobObject(self.handle, 1):
                import ctypes
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.killpg(child.pid, signal.SIGTERM)

    def __exit__(self, *_):
        if self.handle: self.k.CloseHandle(self.handle)


@contextmanager
def experiment_lock(logdir, job_id):
    logdir.mkdir(parents=True, exist_ok=True)
    lock = logdir/'.advanced-training.lock'
    previous = read_json(lock, {})
    if previous and not is_alive(previous.get('pid')):
        require(read_json(lock, {}).get('jobId') == previous.get('jobId'), '锁发生变化，请重查状态。')
        lock.unlink()
    try:
        with lock.open('x', encoding='utf-8') as f: json.dump({'pid': os.getpid(), 'jobId': job_id}, f)
    except FileExistsError:
        raise ValueError('该实验有正在运行的阶段或交接任务，请先查询 status。') from None
    try:
        yield
    finally:
        if read_json(lock, {}).get('jobId') == job_id: lock.unlink(missing_ok=True)


def native_job(spec):
    c, p, exp, repo, logdir, work, sr = context(spec)
    action = spec['action']
    require(action in ['build', 'preprocess', 'run', 'index'], '不支持的后台阶段。')
    work.mkdir(parents=True, exist_ok=True)
    last_progress = 0
    def progress(**extra):
        nonlocal last_progress
        state = read_json(spec['statePath'], {})
        changed = any(state.get(k) != v for k, v in extra.items())
        if changed or time.time() - last_progress > 2:
            write_json(spec['statePath'], {**state, **extra, 'heartbeatAt': time.time()})
            last_progress = time.time()
    with experiment_lock(logdir, spec['jobId']):
        minimum = int(p.get('minSlices') or c.get('minSlices', 1))
        require(minimum >= 1, 'minSlices 必须大于 0。')
        record = read_json(work/'experiment.json', {})
        data = p.get('input') or record.get('input')
        require(data or action != 'build', 'build 需要单角色语音 input。')
        if record: require(record['sampleRate'] == sr and record['repo'] == str(repo), '已有实验的模型路线、采样率或工程不对应。')
        if action == 'build':
            inv = inventory(data, fingerprint=True)
            require(inv['ok'], '输入音频读取失败：'+json.dumps(inv['errors'], ensure_ascii=False))
            source = Path(inv['input'])
            require(not logdir.is_relative_to(source) and not work.is_relative_to(source), '输入不能包含本实验输出目录。')
            if any((logdir/'0_gt_wavs').glob('*.wav')):
                require(p.get('resume') and record.get('fingerprint') == inv['fingerprint'], '已有出料；仅 resume=true 且原始语料指纹一致时复用。')
            write_json(work/'experiment.json', {'experiment': exp, 'repo': str(repo), 'sampleRate': sr, 'input': inv['input'], 'fingerprint': inv['fingerprint'], 'sourceFiles': inv['count']})
        else:
            require(record, '未找到本入口的语料记录，请先用新实验名完成 build；不会接管未知实验。')
            data_proof(logdir, sr, minimum)
        if action in ['run', 'index']: feature_proof(logdir, repo, sr)
        if action == 'run':
            existing = list(logdir.glob('G_*.pth')) + list(logdir.glob('D_*.pth')) + model_candidates(repo, exp)
            require(not existing or p.get('resume'), '已有检查点或小模型，续训请显式 resume=true；不会覆盖重开。')
            if (logdir/'config.json').is_file(): require(read_json(logdir/'config.json')['data']['sampling_rate'] == (48000 if sr == '48k' else 40000), '已有训练配置采样率不同。')
        backup_dir, resume_proof_path, resume_proof = None, None, None
        target_epoch = int(p.get('epochs') or c.get('epochs', 150))
        if action == 'run' and p.get('resume'):
            backup_dir, backup = backup_resume(logdir, work, repo, spec['jobId'], target_epoch, progress)
            runner, resume_proof_path = guarded_runner(repo, Path(spec['statePath']).parent, backup_dir, backup, Path(__file__).parent)
        steps = {'build': 'prep', 'preprocess': 'f0,feat,list', 'run': 'train', 'index': 'index'}[action]
        args = [sys.executable, str(KIT/'scripts/train/p1_run.py'), '--exp', exp, '--steps', steps, '--sr', sr, '--version', 'v2',
                '--min-slices', str(minimum), '--np', str(c.get('processes', 2)), '--f0method', c.get('f0Method', 'rmvpe'),
                '--epochs', str(p.get('epochs') or c.get('epochs', 150)), '--bs', str(p.get('batchSize') or c.get('batchSize', 4)),
                '--se', str(p.get('saveEvery') or c.get('saveEvery', 10))]
        if data: args += ['--data', str(Path(data).resolve())]
        if p.get('resume'): args += ['--resume']
        if resume_proof_path:
            args = [sys.executable, '-u', str(runner), '-e', exp, '-sr', sr, '-f0', '1',
                    '-bs', str(p.get('batchSize') or c.get('batchSize', 4)), '-g', '0',
                    '-te', str(target_epoch), '-se', str(p.get('saveEvery') or c.get('saveEvery', 10)),
                    '-pg', f'assets/pretrained_v2/f0G{sr}.pth', '-pd', f'assets/pretrained_v2/f0D{sr}.pth',
                    '-l', '1', '-c', '0', '-sw', '1', '-v', 'v2']
        environment = {**os.environ, 'RVCX_ROOT': str(work), 'RVCX_REPO': str(repo), 'RVCX_PY': sys.executable, 'RVCX_WORK': str(work), 'RVCX_MODELS': c['modelDir'], 'RVCX_DATASET': str(Path(data).resolve().parent) if data else str(work/'dataset'), 'CUDA_VISIBLE_DEVICES': '0'}
        log = Path(spec['logPath'])
        if Path(spec['stopPath']).exists(): return {'ok': False, 'state': 'stopped', 'stage': action}
        with OwnedProcessTree() as tree, log.open('a', encoding='utf-8', buffering=1) as stream:
            child = subprocess.Popen(args, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), start_new_session=os.name != 'nt')
            tree.attach(child)
            launched_at = time.time()
            while child.poll() is None:
                if resume_proof_path:
                    resume_proof = read_json(resume_proof_path, {})
                    if resume_proof.get('confirmed'):
                        progress(phase='training', resume=resume_proof)
                    elif time.time() - launched_at > 120:
                        tree.terminate(child)
                        child.wait(timeout=30)
                        raise ValueError(f'120 秒内未确认 G/D 和优化器恢复，已停止；不会继续未知起点的训练。备份：{backup_dir}')
                progress()
                if Path(spec['stopPath']).exists():
                    tree.terminate(child)
                    child.wait(timeout=30)
                    return {'ok': False, 'state': 'stopped', 'stage': action, 'exitCode': child.returncode}
                time.sleep(.25)
            require(child.returncode == 0, f'上游 {steps} 失败，退出码 {child.returncode}，见 {log}')
        if action == 'build':
            proof = data_proof(logdir, sr, minimum)
            write_json(work/'experiment.json', {**proof, 'experiment': exp, 'repo': str(repo), 'sampleRate': sr, 'input': inv['input'], 'fingerprint': inv['fingerprint'], 'sourceFiles': inv['count']})
        elif action == 'preprocess': proof = feature_proof(logdir, repo, sr)
        elif action == 'index': proof = index_proof(logdir)
        else:
            if resume_proof_path:
                resume_proof = read_json(resume_proof_path, {})
                require(resume_proof.get('confirmed') and resume_proof.get('jobId') == spec['jobId']
                        and resume_proof.get('fromEpoch') == backup['fromEpoch'], f'没有本次 G/D 实际恢复证明，不能认定续训成功。备份：{backup_dir}')
            models = model_candidates(repo, exp)
            require(models, '训练进程结束但没有导出本实验小模型。')
            proof = inspect_model(models[-1], sr)
            epoch = int(re.search(r'_e(\d+)_s', models[-1].name)[1])
            require(epoch >= int(p.get('epochs') or c.get('epochs', 150)), '模型导出 epoch 未达到请求目标，不能判训练完成。')
            require(list(logdir.glob('G_*.pth')) and list(logdir.glob('D_*.pth')), '缺少可续训的 G/D 检查点。')
        return {'ok': True, 'state': 'succeeded', 'stage': action, 'artifacts': proof, 'qualityAccepted': False,
                **({'backupDir': str(backup_dir), 'resume': resume_proof} if resume_proof_path else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request')
    ap.add_argument('--job')
    a = ap.parse_args()
    if a.job:
        spec = read_json(a.job)
        base = {'jobId': spec['jobId'], 'pid': os.getpid(), 'action': spec['action'], 'startedAt': time.time()}
        write_json(spec['statePath'], {**base, 'ok': True, 'state': 'running'})
        try: result = native_job(spec)
        except Exception as e: result = {'ok': False, 'state': 'failed', 'error': str(e)}
        write_json(spec['statePath'], {**read_json(spec['statePath'], {}), **base, **result, 'finishedAt': time.time()})
    else:
        spec = json.loads(a.request)
        try:
            if spec['action'] == 'doctor': result = doctor(spec)
            elif spec['action'] == 'scan': result = inventory(spec['params']['input'])
            elif spec['action'] == 'register': result = register(spec)
            else: raise ValueError('不支持的前台操作。')
        except Exception as e: result = {'ok': False, 'error': str(e)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

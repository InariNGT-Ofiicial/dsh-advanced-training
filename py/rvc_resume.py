"""Independent, verified resume snapshots and fail-closed RVC checkpoint loading."""
from pathlib import Path
import ast
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4*1024*1024), b''): h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name+f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def checkpoint_metadata(path, load=None):
    if load is None:
        import torch
        load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    d = load(path)
    epoch = d.get('iteration') if isinstance(d, dict) else None
    require(type(epoch) is int and epoch >= 1, f'检查点缺少有效训练轮次：{path}')
    require(isinstance(d.get('model'), dict) and d['model'], f'不是可续训的 G/D 检查点：{path}')
    opt = d.get('optimizer')
    require(isinstance(opt, dict) and opt.get('param_groups') and opt.get('state'), f'缺少优化器状态，不能当作完整续训：{path}')
    lr = d.get('learning_rate')
    require(isinstance(lr, (int, float)) and math.isfinite(lr) and lr > 0, f'检查点学习率无效：{path}')
    return {'epoch': epoch, 'learningRate': lr}


def backup_resume(logdir, work, repo, job_id, target_epoch, progress=lambda **_: None, inspect=checkpoint_metadata):
    require(re.fullmatch(r'[A-Za-z0-9_-]+', job_id), '备份任务 ID 不能是路径。')
    logdir, work, repo = map(lambda p: Path(p).resolve(), [logdir, work, repo])
    target = work/'checkpoint-backups'/(time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+job_id)
    require(not target.is_relative_to(repo/'logs'), '续训备份必须位于引擎 logs 目录之外，请调整 workRoot。')
    selected = []
    for kind in ['G', 'D']:
        paths = list(logdir.glob(kind+'_*.pth'))
        require(paths, f'续训缺少 {kind}_*.pth；不会自动从零训练，请找回成对 G/D 或换新实验名。')
        require(all(re.fullmatch(kind+r'_\d+\.pth', p.name) for p in paths), '检查点命名不是当前训练器支持的 G/D_<数字>.pth。')
        paths.sort(key=lambda p: int(p.stem.split('_')[1]))
        selected.append(paths[-1])
    require(selected[0].stem[2:] == selected[1].stem[2:], '最新 G/D 编号不配对，可能写盘中断；不会静默换成旧轮次。')
    selected += [logdir/'config.json', logdir/'filelist.txt']
    for p in selected:
        require(p.is_file() and p.resolve().parent == logdir and p.stat().st_size > 0, f'缺少有效实验文件或文件指向实验外：{p}')
    required_bytes = sum(p.stat().st_size for p in selected)
    work.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(work).free >= required_bytes + 16*1024*1024, '空间不足，未完成 G/D 备份；本次续训不会启动。')
    target.mkdir(parents=True, exist_ok=False)
    manifest = {'format': 1, 'jobId': job_id, 'repo': str(repo), 'experiment': logdir.name,
                'logDir': str(logdir), 'targetEpoch': target_epoch, 'complete': False, 'files': []}
    save(target/'backup.json', manifest)
    progress(phase='checkpoint-backup', backupDir=str(target))
    for p in selected:
        before = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as src, (target/p.name).open('xb') as dst:
            for data in iter(lambda: src.read(4*1024*1024), b''):
                dst.write(data)
                h.update(data)
                progress(phase='checkpoint-backup', backupDir=str(target))
            dst.flush()
            os.fsync(dst.fileno())
        after = p.stat()
        digest = h.hexdigest()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                and sha(target/p.name) == digest and sha(p) == digest, f'备份期间文件变化或校验失败：{p}；备份目录 {target}')
        manifest['files'].append({'file': p.name, 'source': str(p), 'bytes': after.st_size, 'sha256': digest})
        save(target/'backup.json', manifest)
    epochs = [inspect(target/p.name)['epoch'] for p in selected[:2]]
    require(epochs[0] == epochs[1], f'G/D 实际训练轮次不一致：{epochs}；保留备份但不启动续训：{target}')
    require(target_epoch > epochs[0], f'目标总轮次 {target_epoch} 必须大于已有轮次 {epochs[0]}；备份保留于 {target}')
    manifest.update(complete=True, fromEpoch=epochs[0])
    save(target/'backup.json', manifest)
    progress(phase='resume-ready', backupDir=str(target), resume={'fromEpoch': epochs[0], 'targetEpoch': target_epoch, 'confirmed': False})
    return target, manifest


def guarded_runner(repo, job_dir, backup_dir, manifest, python_dir):
    """Freeze the engine entry file locally; never edit the installed engine."""
    repo, job_dir = Path(repo).resolve(), Path(job_dir).resolve()
    source = (repo/'train/train.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    require({'utils.load_checkpoint', 'utils.latest_checkpoint_path'} <= calls, '该引擎恢复调用不在已支持布局内，拒绝无保护续训。')
    require(not any(isinstance(n, ast.Name) and n.id == '__file__' for n in ast.walk(tree)), '该训练入口依赖相对文件位置，尚未支持安全续训。')
    job_dir.mkdir(parents=True, exist_ok=True)
    frozen = job_dir/'engine-train.py'
    frozen.write_text(source, encoding='utf-8')
    guard = job_dir/'resume-guard.json'
    save(guard, {**manifest, 'backupDir': str(Path(backup_dir).resolve()), 'proofPath': str(job_dir/'resume-proof.json'), 'engineSourceSha256': sha(frozen)})
    runner = job_dir/'resume-train.py'
    runner.write_text('import sys\nfrom pathlib import Path\n'
        f'sys.path.insert(0, {str(repo)!r})\nsys.path.insert(0, {str(Path(python_dir).resolve())!r})\n'
        'from rvc_resume import install_guard\n'+f'install_guard({str(guard)!r})\n'
        f'exec(compile(Path({str(frozen)!r}).read_text(encoding="utf-8"), {str(frozen)!r}, "exec"), globals())\n', encoding='utf-8')
    return runner, job_dir/'resume-proof.json'


class ResumeAbort(BaseException):
    """Bypass the upstream 'except Exception: restart from pretrained' fallback."""


def install_guard(path, utils=None, load=None):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not cfg.get('complete'): raise ResumeAbort('续训备份未完成，拒绝启动。')
    if utils is None:
        from train import utils
    if load is None:
        import torch
        load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    pairs = {x['file'][0]: x for x in cfg['files'] if re.fullmatch(r'[GD]_\d+\.pth', x['file'])}
    loaded = {}

    def latest(directory, regex='G_*.pth'):
        if Path(directory).resolve() != Path(cfg['logDir']).resolve() or regex not in ['G_*.pth', 'D_*.pth']:
            raise ResumeAbort('恢复目标与备份实验不对应，已停止。')
        return str(Path(cfg['backupDir'])/pairs[regex[0]]['file'])

    def restore(checkpoint_path, model, optimizer=None, load_opt=1):
        try:
            kind = Path(checkpoint_path).name[0]
            entry = pairs[kind]
            expected = Path(cfg['backupDir'])/entry['file']
            require(Path(checkpoint_path).resolve() == expected.resolve() and sha(expected) == entry['sha256'], '续训快照路径或 SHA-256 不一致。')
            d = load(expected)
            require(d['iteration'] == cfg['fromEpoch'], '恢复轮次与备份不一致。')
            require(optimizer is not None and load_opt == 1, '必须恢复优化器状态。')
            bare = model.module if hasattr(model, 'module') else model
            current = bare.state_dict()
            require(current.keys() == d['model'].keys() and all(current[k].shape == d['model'][k].shape for k in current), '检查点结构/维度与训练模型不一致。')
            bare.load_state_dict(d['model'], strict=True)
            optimizer.load_state_dict(d['optimizer'])
            loaded[kind] = {'file': str(expected), 'sha256': entry['sha256'], 'epoch': d['iteration'], 'optimizerRestored': True}
            proof = {'jobId': cfg['jobId'], 'pid': os.getpid(), 'fromEpoch': cfg['fromEpoch'], 'targetEpoch': cfg['targetEpoch'], 'backupDir': cfg['backupDir'], 'confirmed': set(loaded) == {'G', 'D'}, 'loaded': loaded}
            save(Path(cfg['proofPath']), proof)
            print('[dsh-resume] '+json.dumps(proof, ensure_ascii=False), flush=True)
            return model, optimizer, d['learning_rate'], d['iteration']
        except Exception as e:
            raise ResumeAbort(f'续训恢复失败，已停止，禁止回退第 1 轮：{e}；备份：{cfg["backupDir"]}') from None

    utils.latest_checkpoint_path = latest
    utils.load_checkpoint = restore

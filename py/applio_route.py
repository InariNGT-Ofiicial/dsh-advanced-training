"""Headless Applio training adapter; fixed interfaces, owned jobs, verified resume."""
from pathlib import Path
import importlib.util
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from rvc_route import (write_json, read_json, require, sha, inventory, experiment_lock,
                       OwnedProcessTree, is_alive)
from rvc_resume import backup_resume, guarded_runner, checkpoint_metadata

CONTRACT = Path(__file__).resolve().parents[1]/'vendor/applio-interface.json'


def context(spec):
    c, p = spec['config'], spec.get('params', {})
    exp = p.get('expName') or c['expName']
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', exp), '实验名必须是安全 ASCII 名称，不能是路径。')
    rate = p.get('sampleRate') or c['sampleRate']
    require(rate == c['sampleRate'], '请在 pipelines.applio 中同时配置采样率和对应底模后重新加载；不在阶段中切换采样率。')
    sr = int(rate[:-1])*1000
    require(sr in ([32000,40000,48000] if c['vocoder'] == 'HiFi-GAN' else [24000,32000]), '采样率与声码器不匹配。')
    repo, logs = Path(c['repo']).resolve(), Path(c['logsDir']).resolve()
    return c, p, exp, repo, logs/exp, Path(c['workRoot']).resolve()/exp, sr


def interfaces(repo):
    contract = read_json(CONTRACT)
    rows = [{'name': name, 'ok': (repo/name).is_file() and hashlib.sha256((repo/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == digest, 'detail': str(repo/name)} for name,digest in contract['files'].items()]
    return contract, rows


def require_interfaces(repo):
    contract, checks = interfaces(repo)
    require(all(x['ok'] for x in checks), 'Applio 训练接口与已核验版本不一致；请先运行 doctor 并使用文档所列版本，不会猜测位置参数或启用 cleanup。')
    return contract


def data_proof(logdir, sr, minimum=1):
    import soundfile as sf
    paths = sorted((logdir/'sliced_audios').glob('*.wav'))
    require(len(paths) >= minimum, f'有效切片不足：{len(paths)} < {minimum}')
    for path in paths:
        i = sf.info(path)
        require(i.frames > 0 and i.channels == 1 and i.samplerate == sr, f'切片规格不正确：{path}')
    return {'slices': len(paths), 'sampleRate': sr}


def feature_proof(logdir, sr, minimum=1):
    import numpy as np
    proof = data_proof(logdir, sr, minimum)
    audio = sorted((logdir/'sliced_audios').glob('*.wav'))
    for path in audio:
        for folder, name in [('extracted',path.stem+'.npy'),('f0',path.name+'.npy'),('f0_voiced',path.name+'.npy')]:
            file = logdir/folder/name
            require(file.is_file(), f'缺少配对特征：{file}')
            a = np.load(file, allow_pickle=False)
            require(a.size > 0 and np.isfinite(a).all(), f'特征为空或非有限数值：{file}')
            if folder == 'extracted': require(a.ndim == 2 and a.shape[1] == 768, '只接受 v2 的 768 维特征。')
    cfg = read_json(logdir/'config.json', {})
    require(cfg.get('data',{}).get('sample_rate') == sr, 'Applio 训练配置采样率不对应。')
    info = read_json(logdir/'model_info.json', {})
    require(info.get('speakers_id') == 1, '本路线接入单角色训练，model_info 的 speakers_id 必须为 1。')
    listing = logdir/'filelist.txt'
    require(listing.is_file(), '缺少 filelist.txt')
    rows = [x for x in listing.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    seen = set()
    for row in rows:
        fields = row.split('|')
        require(len(fields) == 5 and fields[4] == '0', '必须是单角色、5 列训练清单。')
        for value in fields[:4]: require(Path(value).is_file() and Path(value).stat().st_size > 0, f'清单产物不存在：{value}')
        seen.add(Path(fields[0]).resolve())
    require(all(path.resolve() in seen for path in audio) and len(rows) >= len(audio)+2, '清单缺少目标切片或两条静音项。')
    return {**proof, 'featureDim': 768, 'listRows': len(rows), 'filelist': str(listing)}


def model_candidates(logdir, exp):
    pattern = re.compile(re.escape(exp)+r'_(\d+)e_(\d+)s\.pth')
    return sorted([p for p in logdir.glob(exp+'_*.pth') if pattern.fullmatch(p.name)], key=lambda p: tuple(map(int,pattern.fullmatch(p.name).groups())))


def model_proof(path, sr, vocoder):
    import torch
    model = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(model, dict) and isinstance(model.get('weight'),dict) and model['weight'], '不是 Applio 导出音色模型。')
    require(model.get('version') == 'v2' and model.get('f0') == 1 and model.get('sr') == sr and model['config'][-1] == sr, '模型版本/F0/采样率不对应。')
    require(model.get('vocoder') == vocoder and model.get('speakers_id') == 1, '模型声码器或说话人数不对应。')
    require(type(model.get('epoch')) is int and model['epoch'] > 0, '没有有效训练轮次。')
    require(all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in model['weight'].values()), '模型含无效权重。')
    return {'file': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'epoch': model['epoch'], 'step': model.get('step'), 'sampleRate': sr, 'vocoder': vocoder, 'version': 'v2'}


def index_proof(logdir, exp):
    import numpy as np
    import faiss
    path = logdir/(exp+'.index')
    require(path.is_file(), '没有 Applio 成品检索索引。')
    index = faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8).copy())
    require(index.d == 768 and index.ntotal > 0, '成品索引为空或维度错误。')
    return {'file': str(path), 'sha256': sha(path), 'dimensions': index.d, 'vectors': index.ntotal}


def inactive_external(logdir):
    value = read_json(logdir/'config.json', {}).get('process_pids', [])
    if isinstance(value, dict): value = list(value.values())
    require(isinstance(value,list) and not any(type(x) is int and is_alive(x) for x in value), '实验记录中有存活的外部训练进程，请先停止原任务；不接管其进程。')


def doctor(spec):
    c,p,exp,repo,logdir,work,sr = context(spec)
    contract, checks = interfaces(repo)
    def check(name, ok, detail): checks.append({'name': name, 'ok': bool(ok), 'detail': str(detail)})
    for name in ['numpy','soundfile','torch','torchaudio','faiss','librosa','soxr','transformers','tensorboard','matplotlib','sklearn','onnxruntime','torchcrepe']:
        check('module '+name, importlib.util.find_spec(name) is not None, name)
    if not p.get('resumeOnly'):
        for name,value in [('pretrained G',c['pretrainedG']),('pretrained D',c['pretrainedD'])]:
            path = Path(value); check(name,path.is_file() and path.stat().st_size > 0,path)
    for rel in [f'rvc/configs/{sr}.json','assets/config_template.json']:
        check(rel,(repo/rel).is_file(),repo/rel)
    embed = Path(c['embedderDir'])
    if not p.get('resumeOnly'): check('offline embedder', (embed/'config.json').is_file() and any((embed/x).is_file() for x in ['pytorch_model.bin','model.safetensors']), embed)
    predictor = {'rmvpe':'rmvpe.pt','fcpe':'fcpe.pt','swift':'swift.onnx'}.get(c['f0Method'])
    if predictor and not p.get('resumeOnly'): check('F0 model',(repo/'rvc/models/predictors'/predictor).is_file(),predictor)
    for rel in [f'sliced_audios/mute{sr}.wav','extracted/mute.npy','f0/mute.wav.npy','f0_voiced/mute.wav.npy']:
        if not p.get('resumeOnly'): check('mute '+rel,(repo/'logs/mute'/rel).is_file(),repo/'logs/mute'/rel)
    check('backup outside logs',not (work/'checkpoint-backups').is_relative_to(logdir.parent),work/'checkpoint-backups')
    if p.get('deep'):
        import torch
        check('CUDA GPU 0',torch.cuda.is_available(),torch.__version__)
    return {'ok': all(x['ok'] for x in checks), 'checks': checks, 'engine': 'applio', 'sourceCommit': contract['commit'], 'repo':str(repo),'logsDir':str(logdir.parent),'python':sys.executable}


def experiments(spec):
    c,p,exp,repo,logdir,work,sr=context(spec)
    root=logdir.parent
    paths=sorted([x for x in root.iterdir() if x.is_dir() and not x.name.startswith('mute') and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',x.name)]) if root.is_dir() else []
    rows=[{'expName':x.name,'slices':len(list((x/'sliced_audios').glob('*.wav'))),'hasG':any(x.glob('G_*.pth')),'hasD':any(x.glob('D_*.pth'))} for x in paths[:200]]
    return {'ok':True,'stage':'applio-experiments','logsDir':str(root),'experiments':rows,'truncated':len(paths)>200}


def record_for(c,repo,logdir,sr,source):
    return {'engine':'applio','repo':str(repo),'logsDir':str(logdir.parent),'sampleRate':sr,'vocoder':c['vocoder'],'precision':c['precision'], **source}


def record_check(c,repo,logdir,work,sr):
    record = read_json(work/'experiment.json',{})
    require(record and all(record.get(k) == v for k,v in record_for(c,repo,logdir,sr,{}).items()), '实验身份、工程、logsDir、采样率、声码器或精度不对应；已有外部实验请先 adopt。')
    return record


def adopt(spec):
    c,p,exp,repo,logdir,work,sr = context(spec)
    require_interfaces(repo)
    with experiment_lock(logdir,'adopt-'+uuid.uuid4().hex):
        inactive_external(logdir)
        proof = feature_proof(logdir,sr,c['minSlices'])
        epochs=[]; selected=[]
        for kind in ['G','D']:
            paths=sorted(logdir.glob(kind+'_*.pth'),key=lambda f:int(f.stem[2:]) if f.stem[2:].isdigit() else -1)
            require(paths and paths[-1].stem[2:].isdigit(),'已有实验缺少命名正确的成对 G/D。')
            selected.append(paths[-1]);epochs.append(checkpoint_metadata(paths[-1])['epoch'])
        require(selected[0].stem[2:] == selected[1].stem[2:] and epochs[0] == epochs[1], '已有 G/D 编号或轮次不配对。')
        models=model_candidates(logdir,exp)
        if models: model_proof(models[-1],sr,c['vocoder'])
        inv=inventory(logdir/'sliced_audios',fingerprint=True)
        work.mkdir(parents=True,exist_ok=True)
        write_json(work/'experiment.json',record_for(c,repo,logdir,sr,{'adopted':True,'input':inv['input'],'fingerprint':inv['fingerprint'],'sourceFiles':inv['count']}))
    return {'ok':True,'stage':'applio-adopt','fromEpoch':epochs[0],'logsDir':str(logdir.parent),'artifacts':proof,'rendered':f'已接管实验记录 {exp}，检查点第 {epochs[0]} 轮；尚未启动。继续训练使用 applio/resume，自动备份并核验恢复。'}


def register(spec):
    c,p,exp,repo,logdir,work,sr=context(spec)
    with experiment_lock(logdir,'register-'+uuid.uuid4().hex):
        record=record_check(c,repo,logdir,work,sr)
        inactive_external(logdir)
        models=model_candidates(logdir,exp);require(models,'没有本实验导出模型。')
        model=Path(p.get('model') or models[-1]).resolve()
        require(model in [m.resolve() for m in models],'不能登记其他实验的模型。')
        detail=model_proof(model,sr,c['vocoder'])
        idx=index_proof(logdir,exp) if p.get('includeIndex',True) else None
        feature_proof(logdir,sr)
        training=read_json(work/'last-training.json',{})
        require(training.get('logPath') and Path(training['logPath']).is_file(),'缺少本入口训练成功记录与日志；先完成训练/续训。')
        target=Path(p.get('outDir') or Path(c['modelDir'])/exp).resolve()
        require(not target.exists(),f'不覆盖已有音色包：{target}')
        temp=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex);temp.mkdir(parents=True)
        files=[(model,'model.pth'),(logdir/'config.json','config.json'),(logdir/'filelist.txt','filelist.txt'),(logdir/'model_info.json','model_info.json'),(Path(training['logPath']),'training.log')]
        if idx: files.append((Path(idx['file']),exp+'.index'))
        if training.get('resumeProof'): files.append((Path(training['resumeProof']),'resume-proof.json'))
        artifacts=[]
        for src,name in files:
            shutil.copy2(src,temp/name);artifacts.append({'file':name,'sha256':sha(temp/name),'bytes':(temp/name).stat().st_size})
        manifest={'format':1,'route':'applio','artifact_kind':'rvc_voice_model','role':p.get('role') or exp,'model':{**detail,'file':'model.pth'},'index':({**idx,'file':exp+'.index'} if idx else None),'source':record,'interfaceContract':read_json(CONTRACT),'training':training,'artifacts':artifacts,'qualityAccepted':False}
        write_json(temp/'.applio-manifest.json',manifest);os.rename(temp,target)
    return {'ok':True,'stage':'applio-register','outDir':str(target),'manifest':str(target/'.applio-manifest.json'),'artifacts':artifacts,'qualityAccepted':False}


def index_runner(repo,job_dir,logdir,exp):
    job_dir=Path(job_dir);job_dir.mkdir(parents=True,exist_ok=True)
    source=job_dir/'engine-index.py'
    source.write_bytes((repo/'rvc/train/process/extract_index.py').read_bytes())
    target=(logdir/(exp+'.index')).resolve()
    runner=job_dir/'index-run.py'
    runner.write_text('import os,faiss\nfrom pathlib import Path\n'
        f'EXPECTED=Path({str(target)!r})\n'
        'native_write=faiss.write_index\n'
        'def write_index(index,path,*args):\n'
        ' if Path(path).resolve()!=EXPECTED: raise ValueError("Unexpected index target")\n'
        ' temp=EXPECTED.with_name(EXPECTED.name+".tmp-index")\n'
        ' writer=faiss.VectorIOWriter()\n'
        ' native_write(index,writer)\n'
        ' temp.write_bytes(faiss.vector_to_array(writer.data).tobytes())\n'
        ' os.replace(temp,EXPECTED)\n'
        'faiss.write_index=write_index\n'
        f'exec(compile(Path({str(source)!r}).read_text(encoding="utf-8-sig"),{str(source)!r},"exec"),globals())\n',encoding='utf-8')
    return runner


def command_for(action,c,exp,repo,logdir,sr,params,runner=None):
    py=sys.executable
    if action == 'build': return [py,'-u',str(repo/'rvc/train/preprocess/preprocess.py'),str(logdir),str(Path(params['input']).resolve()),str(sr),str(c['processes']),c['cutMode'],'False','3.7','0.3',c['normalization']]
    if action == 'preprocess': return [py,'-u',str(repo/'rvc/train/extract/extract.py'),str(logdir),c['f0Method'],str(c['processes']),'0',str(sr),'custom',str(Path(c['embedderDir']).resolve()),'2']
    if action == 'index': return [py,'-u',str(runner or repo/'rvc/train/process/extract_index.py'),str(logdir),'Auto']
    require(action == 'run','不支持该后台阶段。')
    return [py,'-u',str(runner or repo/'rvc/train/train.py'),exp,str(params.get('saveEvery') or c['saveEvery']),str(params.get('epochs') or c['epochs']),str(Path(c['pretrainedG']).resolve()),str(Path(c['pretrainedD']).resolve()),'0',str(params.get('batchSize') or c['batchSize']),str(sr),'True','True','False','False',c['vocoder'],'False']


def native_job(spec):
    c,p,exp,repo,logdir,work,sr=context(spec);action=spec['action']
    require(action in ['build','preprocess','run','index'],'不支持后台操作。')
    require_interfaces(repo)
    work.mkdir(parents=True,exist_ok=True)
    last=0
    def progress(**extra):
        nonlocal last
        state=read_json(spec['statePath'],{})
        if any(state.get(k)!=v for k,v in extra.items()) or time.time()-last>2:
            write_json(spec['statePath'],{**state,**extra,'heartbeatAt':time.time()});last=time.time()
    with experiment_lock(logdir,spec['jobId']):
        inactive_external(logdir)
        old=read_json(work/'experiment.json',{})
        existing=list(logdir.glob('G_*.pth'))+list(logdir.glob('D_*.pth'))+model_candidates(logdir,exp)
        minimum=p.get('minSlices') or c['minSlices']
        skip=False
        if action == 'build':
            inv=inventory(p['input'],fingerprint=True);require(inv['ok'],'输入音频不能完整读取。')
            source=Path(inv['input'])
            require(not logdir.is_relative_to(source) and not work.is_relative_to(source),'输入不能包含本实验输出目录。')
            if list((logdir/'sliced_audios').glob('*.wav')):
                record_check(c,repo,logdir,work,sr)
                require(p.get('resume') and old.get('fingerprint') == inv['fingerprint'],'已有出料，仅显式 resume 且语料指纹一致时复用。')
                skip=True
            require(not existing or skip,'已有权重，不能覆盖出料。')
            write_json(work/'experiment.json',record_for(c,repo,logdir,sr,{'input':inv['input'],'fingerprint':inv['fingerprint'],'sourceFiles':inv['count']}))
        else:
            record_check(c,repo,logdir,work,sr)
            data_proof(logdir,sr,minimum)
        if action in ['run','index']: feature_proof(logdir,sr,minimum)
        if action == 'preprocess':
            require(not existing,'已有权重，拒绝重写其训练特征。')
            embed=Path(c['embedderDir'])
            require((embed/'config.json').is_file() and any((embed/x).is_file() for x in ['pytorch_model.bin','model.safetensors']), '显式 embedderDir 缺少本地配置或权重；不会回退到其他 embedder。')
            predictor={'rmvpe':'rmvpe.pt','fcpe':'fcpe.pt','swift':'swift.onnx'}.get(c['f0Method'])
            if predictor: require((repo/'rvc/models/predictors'/predictor).is_file(),f'缺少本地 F0 模型 {predictor}，不会自动下载。')
        target_epoch=p.get('epochs') or c['epochs']
        runner=proof_path=backup_dir=backup=None
        if action == 'run':
            require(not existing or p.get('resume'),'已有模型/检查点，继续训练请用 applio/resume；不会重开。')
            if not p.get('resume'):
                for key in ['pretrainedG','pretrainedD']: require(Path(c[key]).is_file(),f'缺少新训练底模 {key}，先运行 doctor。')
            if p.get('resume'):
                extra=['model_info.json']+(['training_data.json'] if (logdir/'training_data.json').is_file() else [])
                backup_dir,backup=backup_resume(logdir,work,repo,spec['jobId'],target_epoch,progress,logs_root=logdir.parent,extra_files=extra,engine='applio')
                runner,proof_path=guarded_runner(repo,Path(spec['statePath']).parent,backup_dir,backup,Path(__file__).parent,engine='applio')
        if action == 'index': runner=index_runner(repo,Path(spec['statePath']).parent,logdir,exp)
        args=command_for(action,c,exp,repo,logdir,sr,p,runner)
        settings=Path(spec['statePath']).parent/'settings.json'
        write_json(settings,{'precision':c['precision'],'model_author':'','discord_presence':False})
        env={**os.environ,'APPLIO_ROOT':str(repo),'APPLIO_CODE_ROOT':str(repo),'APPLIO_LOGS_DIR':str(logdir.parent),'APPLIO_CONFIG_FILE':str(settings),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(repo)}
        if Path(spec['stopPath']).exists(): return {'ok':False,'state':'stopped'}
        if not skip:
            with OwnedProcessTree() as tree, Path(spec['logPath']).open('a',encoding='utf-8',buffering=1) as log:
                child=subprocess.Popen(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt')
                tree.attach(child);launched=time.time()
                while child.poll() is None:
                    progress()
                    if proof_path:
                        proof=read_json(proof_path,{})
                        if proof.get('confirmed'): progress(phase='training',resume=proof)
                        elif time.time()-launched>120:
                            tree.terminate(child);child.wait(timeout=30)
                            raise ValueError(f'未在 120 秒内确认实际恢复，已停止。备份：{backup_dir}')
                    if Path(spec['stopPath']).exists():
                        tree.terminate(child);child.wait(timeout=30)
                        return {'ok':False,'state':'stopped','exitCode':child.returncode}
                    time.sleep(.25)
                require(child.returncode==0,f'Applio {action} 失败，退出码 {child.returncode}；见日志。')
        if action=='build': artifacts=data_proof(logdir,sr,minimum)
        elif action=='preprocess': artifacts=feature_proof(logdir,sr,minimum)
        elif action=='index': artifacts=index_proof(logdir,exp)
        else:
            proof=read_json(proof_path,{}) if proof_path else None
            if proof_path: require(proof.get('confirmed') and proof.get('jobId')==spec['jobId'] and proof.get('fromEpoch')==backup['fromEpoch'],'没有本次实际恢复证明，不能判续训成功。')
            models=model_candidates(logdir,exp);require(models,'训练退出但未导出模型。')
            artifacts=model_proof(models[-1],sr,c['vocoder'])
            require(artifacts['epoch']>=target_epoch,'导出模型尚未达到目标轮次。')
            epochs=[]
            for kind in ['G','D']:
                paths=list(logdir.glob(kind+'_*.pth'));require(paths,f'缺少 {kind} 检查点。')
                path=max(paths,key=lambda f:int(f.stem[2:]))
                epochs.append(checkpoint_metadata(path)['epoch'])
            require(epochs[0]==epochs[1] and epochs[0]>=target_epoch,'G/D 检查点不成对或未达到目标轮次。')
            write_json(work/'last-training.json',{'jobId':spec['jobId'],'logPath':spec['logPath'],'epoch':artifacts['epoch'],'resumeProof':str(proof_path) if proof_path else None,'backupDir':str(backup_dir) if backup_dir else None})
        return {'ok':True,'state':'succeeded','stage':action,'artifacts':artifacts,'qualityAccepted':False,**({'resume':proof,'backupDir':str(backup_dir)} if proof_path else {})}


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request');parser.add_argument('--job');args=parser.parse_args()
    if args.job:
        spec=read_json(args.job);base={'jobId':spec['jobId'],'pid':os.getpid(),'action':spec['action'],'startedAt':time.time()}
        write_json(spec['statePath'],{**base,'ok':True,'state':'running'})
        try: result=native_job(spec)
        except Exception as e: result={'ok':False,'state':'failed','error':str(e)}
        write_json(spec['statePath'],{**read_json(spec['statePath'],{}),**base,**result,'finishedAt':time.time()})
    else:
        spec=json.loads(args.request)
        try:
            action=spec['action']
            if action=='doctor': result=doctor(spec)
            elif action=='experiments': result=experiments(spec)
            elif action=='scan': result=inventory(spec['params']['input'])
            elif action=='adopt': result=adopt(spec)
            elif action=='register': result=register(spec)
            else: raise ValueError('不支持前台操作。')
        except Exception as e: result={'ok':False,'error':str(e)}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()

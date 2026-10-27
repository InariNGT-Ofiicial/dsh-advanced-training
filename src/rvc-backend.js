/** RVC v2 training adapter with durable, owned jobs and artifact-checked handoff. */
import { existsSync, statSync, mkdirSync, readFileSync, writeFileSync, renameSync, openSync, readSync, closeSync } from 'node:fs';
import { resolve, join, dirname, delimiter } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { runOnce, spawnJob } from '../vendor/ddsp-pipeline/src/execution.js';
import { lastJson } from './exec.js';

const SCRIPT = fileURLToPath(new URL('../py/rvc_route.py', import.meta.url));
const EXP = '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$';
const schema = (properties, required = []) => ({ type: 'object', properties, required, additionalProperties: false });
const str = description => ({ type: 'string', minLength: 1, description });
const exp = { type: 'string', pattern: EXP, description: 'ASCII 实验名，对应所选训练器的实验目录，角色显示名可另用中文' };
const rate = { type: 'string', enum: ['48k', '40k'] };
const positive = { type: 'integer', minimum: 1 };
const out = (ok, rendered, extra = {}) => ({ ok, rendered, ...extra });
const read = (p, fallback = null) => existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : fallback;
const file = p => !!p && existsSync(p) && statSync(p).isFile();
function atomic(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  const temp = `${path}.${randomUUID()}.tmp`;
  writeFileSync(temp, JSON.stringify(value, null, 2), 'utf8');
  renameSync(temp, path);
}
function tail(path) {
  if (!file(path)) return '';
  const size = statSync(path).size; const n = Math.min(size, 16000); const b = Buffer.alloc(n);
  const fd = openSync(path, 'r');
  try { readSync(fd, b, 0, n, size - n); } finally { closeSync(fd); }
  return b.toString('utf8').split(/\r?\n/).slice(-25).join('\n');
}

export function resolveRvcConfig(raw = {}) {
  const root = raw.root || raw.rvcRoot || process.env.DSH_RVC_ROOT || process.env.RVCX_ROOT || '';
  const repo = raw.repo || raw.rvcRepo || process.env.RVCX_REPO || (root ? join(root, 'repo') : '');
  const python = raw.python || raw.rvcEnv || process.env.RVCX_PY || (root ? join(root, 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python') : '');
  const workRoot = resolve(raw.workRoot || process.cwd(), 'rvc');
  const ffmpegDir = raw.ffmpegDir || raw.rvcFfmpegDir || process.env.RVCX_FFMPEG_DIR || (file(raw.ffmpeg) ? dirname(resolve(raw.ffmpeg)) : '');
  return { root, repo, python, workRoot, ffmpegDir,
    expName: raw.expName || 'my-rvc-voice', modelDir: resolve(raw.modelDir || join(workRoot, 'models')),
    sampleRate: raw.sampleRate || '48k', f0Method: raw.f0Method || 'rmvpe',
    epochs: raw.epochs ?? 150, batchSize: raw.batchSize ?? 4, saveEvery: raw.saveEvery ?? 10,
    processes: raw.processes ?? 2, minSlices: raw.minSlices ?? 1 };
}

export function createRvcBackend(raw = {}, dependencies = {}, profile = {}) {
  const c = (profile.resolveConfig || resolveRvcConfig)(raw);
  const route = profile.route || 'rvc';
  const label = profile.label || 'RVC';
  const script = profile.script || SCRIPT;
  const routeRate = profile.sampleRates ? { type: 'string', enum: profile.sampleRates } : rate;
  const invoke = dependencies.runOnce || runOnce;
  const launch = dependencies.spawnJob || spawnJob;
  const environment = { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1', CUDA_VISIBLE_DEVICES: '0',
    OMP_NUM_THREADS: '1', ...(c.ffmpegDir ? { PATH: c.ffmpegDir + delimiter + (process.env.PATH || ''), RVCX_FFMPEG_DIR: c.ffmpegDir } : {}) };
  const request = (action, params) => ({ action, params, config: c });
  const check = () => {
    if (profile.check) return profile.check(c);
    if (!c.repo || !c.root || !file(c.python)) throw new Error('请在 pipelines.rvc 配置 root/repo/python（或 rvcRoot/rvcRepo/rvcEnv），Python 必须指向已有 RVC 环境。');
    if (!new RegExp(EXP).test(c.expName)) throw new Error('配置中的 expName 必须是 ASCII 实验名，不能是路径。');
    if (!['48k', '40k'].includes(c.sampleRate) || !['pm', 'rmvpe'].includes(c.f0Method)) throw new Error('RVC 本版只支持 v2、48k/40k、pm/rmvpe。');
    for (const k of ['epochs', 'batchSize', 'saveEvery', 'processes', 'minSlices']) if (!Number.isInteger(c[k]) || c[k] < 1) throw new Error(`${k} 必须为正整数。`);
  };
  const foreground = async (action, params) => {
    check();
    const r = await invoke({ cmd: c.python, args: ['-u', script, '--request', JSON.stringify(request(action, params))], env: environment, timeoutMs: action === 'adopt' ? 300000 : 120000 });
    const value = lastJson(r.stdout);
    if (!r.ok || !value) return out(false, `${label} ${action} 执行失败：${r.stderr || r.error || r.stdout || '无结果'}`, { stage: `${route}-${action}`, exitCode: r.exitCode });
    const lines = [`【${label} ${action}】${value.ok ? '通过' : '未通过'}`];
    if (value.error) lines.push(value.error);
    if (value.checks) for (const v of value.checks) lines.push(`${v.ok ? '✓' : '×'} ${v.name}: ${v.detail}`);
    if (action === 'scan') lines.push(`音频 ${value.count || 0} 条，实测 ${(value.seconds || 0).toFixed(2)} 秒；仅用于输入盘点，角色与听感需人工确认。`);
    if (value.outDir) lines.push(`模型、索引、配置、训练清单和日志：${value.outDir}`, '文件交接通过不代表音质验收。');
    if (value.logsDir) lines.push(`实验根目录：${value.logsDir}`);
    if (value.fromEpoch) lines.push(`已有检查点第 ${value.fromEpoch} 轮；继续训练使用 ${route}/resume。`);
    if (value.experiments) lines.push(...value.experiments.map(x => `${x.expName}：切片 ${x.slices}，G/D ${x.hasG && x.hasD ? '齐备，需 adopt 核对实际轮次' : '不完整'}`));
    return { ...value, rendered: lines.join('\n') };
  };
  const jobFiles = params => {
    const name = params.expName || c.expName;
    if (!new RegExp(EXP).test(name)) throw new Error('expName 不能是路径。');
    const work = join(c.workRoot, name);
    const latest = read(join(work, 'latest.json'));
    const id = params.jobId || latest?.jobId;
    if (!id) return { work, name };
    if (!/^[0-9a-f-]{36}$/.test(id)) throw new Error('jobId 不是本入口生成的任务 ID。');
    const dir = join(work, 'jobs', id); const spec = read(join(dir, 'request.json'));
    if (!spec || spec.jobId !== id || spec.params.expName !== name) throw new Error(`找不到本入口拥有的 ${label} 任务。`);
    return { work, name, spec, state: read(spec.statePath, { state: 'queued', ok: true }) };
  };
  const status = async params => {
    const j = jobFiles(params);
    if (!j.spec) return out(true, `${label} ${j.name} 尚无本入口任务；既有外部进程不由本工具接管。`, { stage: `${route}-status`, state: 'not-started' });
    let state = j.state.state;
    const launched = read(join(dirname(j.spec.statePath), 'launch.json'));
    const pid = j.state.pid || launched?.pid;
    if (state === 'running' || (state === 'queued' && pid)) {
      let alive = false;
      try { process.kill(pid, 0); alive = true; } catch (e) { alive = e.code === 'EPERM'; }
      // A fresh heartbeat wins over a transient OS process-probe failure.
      if (!alive && Date.now()/1000 - (j.state.heartbeatAt || j.state.startedAt || 0) > 10) state = 'lost';
      else if (j.state.heartbeatAt && Date.now()/1000 - j.state.heartbeatAt > 60) state = 'unknown';
    }
    const ok = !['failed', 'lost', 'unknown', 'stopped'].includes(state);
    const resume = j.state.resume;
    const recovery = (j.state.backupDir ? `\nG/D 独立备份：${j.state.backupDir}` : '')
      + (resume ? `\n续训起点：第 ${resume.fromEpoch} 轮；目标总轮次 ${resume.targetEpoch}；G/D 与优化器恢复${resume.confirmed ? '已确认' : '待确认'}` : '');
    return out(ok, `【${label} ${j.spec.action}】${state}\n${j.state.error || ''}${recovery}\n日志：${j.spec.logPath}\n${tail(j.spec.logPath)}`,
      { ...j.state, ok, stage: `${route}-status`, state, jobId: j.spec.jobId, logPath: j.spec.logPath, statePath: j.spec.statePath, qualityAccepted: false });
  };
  const background = async (action, params) => {
    check();
    const expName = params.expName || c.expName;
    if (!new RegExp(EXP).test(expName)) throw new Error('expName 必须是安全的 ASCII 实验名。');
    const jobId = randomUUID(); const work = join(c.workRoot, expName); const dir = join(work, 'jobs', jobId);
    const spec = { ...request(action, { ...params, expName }), jobId,
      logPath: join(work, `stage-${action === 'run' ? 'train' : action}.log`),
      statePath: join(dir, 'state.json'), stopPath: join(dir, 'stop.request') };
    mkdirSync(dir, { recursive: true });
    atomic(join(dir, 'request.json'), spec);
    atomic(spec.statePath, { ok: true, state: 'queued', jobId, action, startedAt: Date.now()/1000 });
    const child = await launch({ cmd: c.python, args: ['-u', script, '--job', join(dir, 'request.json')], env: environment, logPath: spec.logPath });
    if (!child.pidValid) {
      atomic(spec.statePath, { ok: false, state: 'failed', jobId, error: '后台 Python 启动失败' });
      return out(false, `${label} 后台任务启动失败：${tail(spec.logPath)}`, { stage: `${route}-${action}`, jobId });
    }
    atomic(join(dir, 'launch.json'), { jobId, pid: child.pid });
    atomic(join(work, 'latest.json'), { jobId, action, expName });
    return out(true, `${label} ${action} 后台任务已启动：${jobId}\n请用 ${route}/status 并传 expName=${expName}、jobId 核查结果；阶段完成前不要推进下一步。\n日志：${spec.logPath}`,
      { stage: `${route}-${action}`, state: 'started', jobId, expName, pid: child.pid, logPath: spec.logPath, statePath: spec.statePath });
  };
  const protect = fn => async (params = {}) => {
    try { return await fn(params); } catch (e) { return out(false, e.message, { stage: route, error: e.message }); }
  };
  const defs = new Map();
  const add = (action, description, properties, required, execute) => defs.set(action, { name: `${route}_${action}`, description: profile.descriptions?.[action] || description, parameters: schema(Object.fromEntries(Object.entries(properties).map(([k,v]) => [k, k === 'sampleRate' ? routeRate : v])), required), execute: protect(execute) });
  add('doctor', '检查 RVC v2 工程、独立 Python、预训练模型、HuBERT、所选 F0 工具、静音样本和子进程 FFmpeg。deep=true 另查 CUDA。', { expName: exp, sampleRate: rate, deep: { type: 'boolean' }, ...(profile.doctorProperties || {}) }, [], p => foreground('doctor', p));
  add('scan', '盘点已人工确认的单角色语音目录，实测时长和音频规格；不修改原始音频，不接受混合角色语料作为已确认输入。', { input: str('单角色原始音频目录') }, ['input'], p => foreground('scan', p));
  add('build', '后台调用 RVC 原生 prep，生成目标采样率和 16k 配对切片；回读数量/规格，不能仅按退出码判断。已有出料仅在 resume=true 且语料指纹一致时复用。', { expName: exp, input: str('单角色音频目录'), sampleRate: rate, minSlices: positive, resume: { type: 'boolean' } }, ['input'], p => background('build', p));
  add('preprocess', '后台执行 F0 → HuBERT v2 特征 → filelist，逐条检查配对、维度和有限数值；必须等 build 成功后调用。', { expName: exp, sampleRate: rate, minSlices: positive }, [], p => background('preprocess', p));
  add('run', '后台训练 RVC v2（GPU 0）；默认拒绝已有检查点。resume=true 必须有成对 G/D 与优化器状态，先备份到引擎 logs 外并核验哈希，再确认实际恢复轮次；备份/恢复失败即停止，禁止回退第 1 轮。epochs 为目标总轮次，必须大于已有轮次。用 status 查看备份位置与恢复证明。', { expName: exp, sampleRate: rate, epochs: positive, batchSize: positive, saveEvery: positive, resume: { type: 'boolean' } }, [], p => background('run', p));
  add('index', '后台构建检索索引，并回读验证 added 索引包含向量且维度为 768；trained 中间索引不算成品。', { expName: exp, sampleRate: rate }, [], p => background('index', p));
  add('status', '读取本入口任务状态、真实日志与经核查的阶段产物；succeeded 只证明该阶段通过，训练状态不能代表听感。', { expName: exp, jobId: str('启动返回的任务 ID，可省略取该实验最新任务') }, [], status);
  add('stop', '只向本入口拥有的任务发送停止请求，由它终止自己的子进程树；不会按实验名搜索并终止其他外部进程。之后用 status 确认 stopped。', { expName: exp, jobId: str('本入口任务 ID') }, [], async p => {
    const j = jobFiles(p);
    if (!j.spec) return out(false, '没有可停止的本入口任务。');
    if (!['queued', 'running'].includes(j.state.state)) return out(true, `任务已结束：${j.state.state}`, { state: j.state.state });
    writeFileSync(j.spec.stopPath, j.spec.jobId, 'utf8');
    return out(true, '已发送停止请求；用同一 jobId 查询 status 确认进程与阶段结果。', { stage: `${route}-stop`, jobId: j.spec.jobId, state: 'stop-requested' });
  });
  add('register', '核验本实验 RVC 小模型、added 索引、配置和日志，归档为独立音色包并记录 SHA-256；拒绝覆盖，G/D 训练检查点不能冒充成品。includeIndex=false 可明确只交接无检索索引的模型。', { expName: exp, sampleRate: rate, role: str('角色显示名，可中文'), model: str('可选：本实验导出的某个小模型'), outDir: str('新的音色包目录'), includeIndex: { type: 'boolean' } }, [], p => foreground('register', p));
  if (profile.adopt) {
    add('experiments', '只读列出配置 logsDir 中的 Applio 实验、切片数和 G/D 是否存在；不会把文件名提示当作恢复轮次。', {}, [], p => foreground('experiments', p));
    add('adopt', '核验配置中 logsDir 下的已有 Applio 单角色实验、特征、G/D 和轮次，再记录为插件实验；不启动训练、不清理权重。外部训练须先停止。', { expName: exp, sampleRate: rate }, [], p => foreground('adopt', p));
    add('resume', '继续训练：自动独立备份 G/D、配置、清单和模型信息，核验哈希并严格恢复模型、优化器与 AMP 状态；任何检查失败即停止，不调用外部界面或 cleanup。epochs 为目标总轮次。', { expName: exp, sampleRate: rate, epochs: positive, batchSize: positive, saveEvery: positive }, [], p => background('run', { ...p, resume: true }));
  }
  return defs;
}

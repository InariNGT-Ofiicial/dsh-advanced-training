/** Unified training entry. Backends stay private; inference is deliberately excluded. */
import { existsSync, statSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { apply as applyDdsp } from '../vendor/ddsp-pipeline/src/index.js';
import { apply as applyGsv } from '../vendor/anime-voice/plugins/tts-train-pipeline/src/index.js';
import { createAnimeBackend } from './anime-backend.js';

const MANIFEST = JSON.parse(readFileSync(new URL('../vendor/manifest.json', import.meta.url), 'utf8'));
const ACTIONS = {
  ddsp: ['sources', 'doctor', 'setup', 'hunt', 'toolchain', 'fetch', 'unpack', 'acquire', 'plan', 'scan', 'build', 'qc', 'preprocess', 'run', 'status', 'stop', 'register', 'overview'],
  gsv: ['doctor', 'plan', 'script', 'scan', 'build', 'asr', 'patch_text', 'preprocess', 'run', 'status', 'stop', 'register', 'overview'],
};
const objectSchema = (properties, required = []) => ({ type: 'object', properties, required, additionalProperties: false });
const str = (description) => ({ type: 'string', minLength: 1, description });
const reply = (ok, rendered, extra = {}) => ({ ok, rendered, ...extra });

/** Validate the backend's real parameter schema before calling its handler directly. */
export function validate(schema, value, path = 'params') {
  const errors = [];
  const t = schema.type;
  const matches = t === 'object' ? value !== null && typeof value === 'object' && !Array.isArray(value)
    : t === 'array' ? Array.isArray(value)
      : t === 'integer' ? Number.isInteger(value)
        : t === 'number' ? typeof value === 'number' && Number.isFinite(value)
          : !t || typeof value === t;
  if (!matches) return [`${path} must be ${t}`];
  if (schema.enum && !schema.enum.includes(value)) errors.push(`${path} must be one of ${schema.enum.join(', ')}`);
  if (typeof value === 'string' && schema.minLength && value.length < schema.minLength) errors.push(`${path} must not be empty`);
  if (t === 'object') {
    for (const name of schema.required || []) if (!(name in value)) errors.push(`${path}.${name} is required`);
    for (const [key, v] of Object.entries(value)) {
      if (['__proto__', 'prototype', 'constructor'].includes(key)) { errors.push(`${path}.${key} is forbidden`); continue; }
      const prop = schema.properties?.[key];
      if (!prop && schema.additionalProperties === false) errors.push(`${path}.${key} is not supported`);
      else if (prop) errors.push(...validate(prop, v, `${path}.${key}`));
    }
  }
  if (t === 'array' && schema.items) value.forEach((v, i) => errors.push(...validate(schema.items, v, `${path}[${i}]`)));
  if (t === 'array' && schema.minItems && value.length < schema.minItems) errors.push(`${path} needs at least ${schema.minItems} items`);
  if (typeof value === 'number' && schema.minimum !== undefined && value < schema.minimum) errors.push(`${path} must be >= ${schema.minimum}`);
  return errors;
}

async function capture(apply, ctx, config, prefix, actions) {
  const definitions = new Map();
  const backendCtx = {
    logger: ctx.logger || console,
    get(service) {
      return service === 'tools' ? { register(def) { definitions.set(def.name, def); } } : ctx.get(service);
    },
  };
  await apply(backendCtx, config);
  const selected = new Map();
  for (const action of actions) {
    const name = `${prefix}_${action}`;
    const def = definitions.get(name);
    if (!def) throw new Error(`Bundled backend is missing ${name}`);
    selected.set(action, def);
  }
  return selected;
}

export function backendConfig(config, route) {
  const common = {};
  const keys = route === 'ddsp'
    ? ['songRoot', 'trainRoot', 'voiceLibDir', 'toolsDir', 'ddspDir', 'ddspEnv', 'ffmpeg', 'expName', 'timeoutMs', 'maxOutputChars']
    : ['songRoot', 'workRoot', 'gsvDir', 'gsvEnv', 'ffmpeg', 'expName', 'lang', 'asrCacheDir', 'timeoutMs', 'maxOutputChars'];
  for (const key of keys) if (config[key] !== undefined && config[key] !== '') common[key] = config[key];
  if (route === 'ddsp' && !common.ddspEnv && config.pyEnv) common.ddspEnv = config.pyEnv;
  return { ...common, ...(config.pipelines?.[route] || {}) };
}

export async function createTrainingHub(ctx, config = {}) {
  const backends = new Map([
    ['ddsp', await capture(applyDdsp, ctx, backendConfig(config, 'ddsp'), 'train', ACTIONS.ddsp)],
    ['gsv', await capture(applyGsv, ctx, backendConfig(config, 'gsv'), 'tts', ACTIONS.gsv)],
    ['anime', createAnimeBackend(config.pipelines?.anime || {})],
  ]);
  const receiptDir = resolve(config.receiptDir || join(config.workRoot || process.cwd(), 'advanced-training', 'receipts'));
  const catalog = ({ route, action } = {}) => {
    if (route && !backends.has(route)) return reply(false, `不支持的路线：${route}`);
    const routes = [];
    for (const [id, defs] of backends) {
      if (route && route !== id) continue;
      if (action && !defs.has(action)) return reply(false, `${id} 不支持操作 ${action}`);
      routes.push({ route: id, kind: id === 'anime' ? 'material-preparation' : 'model-training', actions: [...defs]
        .filter(([a]) => !action || a === action)
        .map(([a, d]) => ({ action: a, backendTool: d.name, description: d.description, parameters: d.parameters })) });
    }
    return reply(true, routes.map(r => `${r.route}: ${r.actions.map(a => a.action).join(' / ')}`).join('\n')
      + '\n通过 adv_train 调用；训练与素材步骤接入，推理、翻唱和发布不接入。RVC 保留原有质量检查能力，未接入训练编排。',
    { stage: 'catalog', routes, sources: MANIFEST.sources.map(({ files, ...s }) => s), receiptDir });
  };
  const execute = async ({ route, action, params = {}, dryRun = false } = {}) => {
    const def = backends.get(route)?.get(action);
    if (!def) return reply(false, `不支持的路线或操作：${route}/${action}。先调用 adv_train_catalog 查看可用操作。`, { stage: 'dispatch', executed: false });
    const errors = validate(def.parameters, params);
    if (errors.length) return reply(false, errors.join('\n'), { stage: 'parameters', errors, executed: false });
    if (dryRun) return reply(true, `仅预览：${route}/${action} → ${def.name}；没有执行，未验证依赖或产物。`,
      { stage: 'dry-run', executed: false, route, action, backendTool: def.name, params });
    if (route === 'gsv' || route === 'ddsp') {
      const resolvedConfig = backendConfig(config, route);
      const key = route === 'gsv' ? 'gsvDir' : 'ddspDir';
      if (resolvedConfig[key] && (!existsSync(resolvedConfig[key]) || !statSync(resolvedConfig[key]).isDirectory())) return reply(false, `显式配置的 ${key} 不是可用目录：${resolvedConfig[key]}。请修正配置；不会回退到另一份工程。`, { stage: 'configuration', executed: false });
    }
    const startedAt = new Date().toISOString();
    let result;
    try {
      result = await def.execute(params);
      if (!result || typeof result.ok !== 'boolean') throw new Error('Backend returned no explicit success/failure');
    } catch (e) {
      result = reply(false, `执行失败：${e.message}`, { error: String(e.message) });
    }
    const out = { ...result, route, action, backendTool: def.name, executed: true };
    const receipt = join(receiptDir, `${Date.now()}-${randomUUID()}.json`);
    try {
      mkdirSync(receiptDir, { recursive: true });
      writeFileSync(receipt, JSON.stringify({ format: 1, startedAt, finishedAt: new Date().toISOString(), route, action, backendTool: def.name, params, result }, null, 2) + '\n', { encoding: 'utf8', flag: 'wx' });
      out.receipt = receipt;
    } catch (e) {
      out.receiptError = String(e.message);
      out.rendered += `\n执行结果已返回，但回执写盘失败：${e.message}。不要据此重复启动训练。`;
    }
    if (action === 'run' && result.ok && !params.dryRun) {
      out.rendered += '\n启动回执不代表训练完成；继续使用同一路线的 status 检查进程、日志和检查点，再交接模型。';
    }
    if (out.receipt) out.rendered += `\n统一入口回执：${out.receipt}`;
    return out;
  };
  return { catalog, execute };
}

export function trainingPlan(args = {}) {
  const { route, sourceType = 'audio', source, speaker = '', linesFile = '', outDir = '' } = args;
  if (!['ddsp', 'gsv'].includes(route) || !source || !['audio', 'game', 'anime'].includes(sourceType)) return reply(false, '需要 route=ddsp/gsv、source 和 sourceType=audio/game/anime。');
  const steps = [];
  const add = (r, action, params, note = '') => steps.push({ tool: 'adv_train', route: r, action, params, note });
  add(route, 'doctor', {}, '只检查选择的训练路线；按报告补齐工程、Python、依赖和预训练权重。');
  let input = source;
  let lines = linesFile;
  if (sourceType === 'game') {
    add('ddsp', 'unpack', { input: source, ...(outDir ? { outDir } : {}) }, '仅兼容的游戏封包；读取返回的实际解包目录，再替换后续 source。');
    input = '<unpack 返回的实际目录>';
    if (route === 'gsv' && !lines) {
      add('gsv', 'script', { voiceDir: '<解包后的语音目录>' }, '优先保留游戏剧本原文；读取返回的 lines.jsonl。');
      lines = '<script 返回的 lines.jsonl>';
    }
  }
  if (sourceType === 'anime') {
    add('anime', 'extract', { input: source, ...(outDir ? { outDir } : {}) }, '提供 outDir 和本地 ASR/声纹模型；导出候选角色与台词，人工确认角色。');
    add('anime', 'refine', { runDir: '<extract 输出目录>', mapFile: '<人工确认的角色映射.json>', target: speaker || '<角色名>', outDir: '<新的干净切片目录>' }, '声纹/ASR 是辅助筛选，需回听角色与台词，输入含配乐时先分离或使用 rebuild 整集重建。');
    input = '<refine 返回的角色语音目录>';
    if (route === 'gsv') {
      add('anime', 'bridge', { input: '<refine 输出的 lines.jsonl>', out: '<新建的训练台词.jsonl>', ...(speaker ? { speaker } : {}) }, '把番剧 file 字段转为训练器的 voice_path，保留原台词；读取桥接后的路径。');
      lines = '<bridge 返回的训练台词.jsonl>';
    }
  }
  if (route === 'ddsp') {
    add(route, 'scan', { input });
    add(route, 'build', { input, speakers: [speaker || '<确认的角色名>'] }, '从返回值选择该角色的实际 datasetDir；保留自然尾音。');
    add(route, 'qc', { dir: '<build 返回的该角色数据集目录>' }, '检查不合格项并处理后再进入训练。');
    add(route, 'preprocess', { datasetDir: '<通过检查的该角色数据集目录>' }, '使用独立 DDSP 工程；上游会备份原有训练音频目录。');
  } else {
    add(route, 'scan', { source: lines || input });
    add(route, 'build', lines ? { lines, ...(speaker ? { speaker } : {}) } : { voiceDir: input, speaker: speaker || '<确认的角色名>', voiceOnly: true, allowEmptyText: true }, lines ? '原文与音频配对后出料。' : '自备无文本音频先规格化，再补标注。');
    if (!lines) add(route, 'asr', { wavDir: '<build 返回的 wav_dir>' }, '只对缺失台词的语料做 ASR；人工校对后才能预处理。');
    steps.push({ tool: 'adv_tail_audit', params: { dir: '<build 返回的 wav_dir>' }, note: '检查句尾完整性；源音频已切尾需回源重建。' });
    add(route, 'preprocess', { listFile: '<已校对的 list.txt>', wavDir: '<对应 wav 目录>' });
  }
  add(route, 'run', {}, '后台启动；不是训练完成。');
  add(route, 'status', {}, '检查进程、日志、检查点，完成后核对模型与配置。');
  add(route, 'register', route === 'ddsp' ? { role: speaker || '<角色名>' } : { name: speaker || '<角色名>' }, '保留该路线的独立模型格式，注册文件不等于音质验收。');
  return reply(true, `【统一训练计划 · ${route} / ${sourceType}】\n${steps.map((s, i) => `${i + 1}. ${s.tool}${s.action ? ` ${s.route}/${s.action}` : ''} ${s.note || ''}`).join('\n')}\n尖括号为上一步实际输出或人工确认的信息，不能原样执行。本工具没有执行任何步骤。`,
    { stage: 'plan', executed: false, route, sourceType, steps, scope: '素材接入、数据整理、质量检查、对应模型训练与模型交接；不调用推理、翻唱或发布。' });
}

export async function registerTrainingHub(ctx, config = {}) {
  const hub = await createTrainingHub(ctx, config);
  const tools = ctx.get('tools');
  const output = { schema: { type: 'object', properties: { ok: { type: 'boolean' }, rendered: { type: 'string' }, stage: { type: 'string' } }, required: ['ok', 'rendered'] }, render: (_a, v) => [{ type: 'text', text: String(v?.rendered ?? '(无输出)') }] };
  tools.register({ name: 'adv_train_catalog', description: '统一训练入口：发现内置 DDSP、GPT-SoVITS 与番剧素材后端；返回真实可调用操作及其参数 schema。无需分别安装训练插件。不包含推理/翻唱/发布。', parameters: objectSchema({ route: { type: 'string', enum: ['ddsp', 'gsv', 'anime'] }, action: str('可选：只看某个操作的参数') }), output, execute: async (a = {}) => hub.catalog(a) });
  tools.register({ name: 'adv_train_plan', description: '从游戏、番剧或自备单人语音集制定逐步训练流程；区分 DDSP 与 GPT-SoVITS 的数据、标注和模型格式。本工具只制定计划；真实执行使用 adv_train。', parameters: objectSchema({ route: { type: 'string', enum: ['ddsp', 'gsv'] }, sourceType: { type: 'string', enum: ['audio', 'game', 'anime'] }, source: str('本地素材路径'), speaker: str('目标角色'), linesFile: str('已有带台词 JSONL，可省略'), outDir: str('解包或番剧提取的新输出目录') }, ['route', 'source']), output, execute: async (a = {}) => trainingPlan(a) });
  tools.register({ name: 'adv_train', description: '统一执行已接入的训练/素材操作。先用 adv_train_catalog 查询 route/action 的真实 params，再调用。route=ddsp/gsv 管训练，anime 管摘句/整集重建/台词桥接。返回原始后端结果与持久回执；run 只表示启动，需 status 核查。禁止推理、翻唱和发布操作。', parameters: objectSchema({ route: { type: 'string', enum: ['ddsp', 'gsv', 'anime'] }, action: str('catalog 中的操作名'), params: { type: 'object', additionalProperties: true, description: '严格按所选操作的参数 schema 填写' }, dryRun: { type: 'boolean', description: '只检查参数并预览派发，不执行，也不验证依赖/产物' } }, ['route', 'action']), output, execute: hub.execute });
  return hub;
}

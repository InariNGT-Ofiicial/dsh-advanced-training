/** Adapt the existing anime scripts to the same typed invocation contract as training. */
import { existsSync, statSync, readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, join, dirname, basename, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runOnce } from '../vendor/ddsp-pipeline/src/execution.js';

const EXTRACTOR = fileURLToPath(new URL('../vendor/anime-voice/skills/anime-voice-line-extractor/scripts/', import.meta.url));
const REBUILDER = fileURLToPath(new URL('../vendor/anime-voice/skills/anime-voice-clone-training/scripts/run_pipeline.py', import.meta.url));
const schema = (properties, required) => ({ type: 'object', properties, required, additionalProperties: false });
const s = (description) => ({ type: 'string', minLength: 1, description });
const file = p => !!p && existsSync(p) && statSync(p).isFile();
const nonempty = p => file(p) && statSync(p).size > 0;
const answer = (ok, rendered, extra = {}) => ({ ok, rendered, ...extra });

function freshDir(path) {
  const p = resolve(path);
  if (existsSync(p) && (!statSync(p).isDirectory() || readdirSync(p).length)) throw new Error(`输出目录必须不存在或为空：${p}`);
  return p;
}

function localModels(c, whisper = true) {
  if (whisper) {
    const root = c.whisperModel;
    const snapshots = root && join(root, 'snapshots');
    const found = root && (file(join(root, 'model.bin')) || (existsSync(snapshots) && readdirSync(snapshots).some(d => file(join(snapshots, d, 'model.bin')))));
    if (!found) throw new Error('请配置 pipelines.anime.whisperModel，指向已下载的本地 faster-whisper 模型目录。');
  }
  if (!c.camppModel || !file(join(c.camppModel, 'campplus_cn_common.bin'))) throw new Error('请配置 pipelines.anime.camppModel，指向含 campplus_cn_common.bin 的本地目录。');
}

/** Bridge the anime extractor's `file` field / native .list into the trainer's voice_path. */
export function bridgeAnime(params) {
  const input = resolve(params.input);
  const out = resolve(params.out);
  if (!file(input)) throw new Error(`清单不存在：${input}`);
  if (existsSync(out)) throw new Error(`不覆盖已有清单：${out}`);
  const base = dirname(input);
  const text = readFileSync(input, 'utf8').replace(/^\uFEFF/, '');
  const isJson = /\.jsonl$/i.test(input);
  const rows = text.split(/\r?\n/).filter(l => l.trim()).map((line, i) => {
    if (isJson) {
      try { return JSON.parse(line); } catch { throw new Error(`第 ${i + 1} 行不是合法 JSON`); }
    }
    const parts = line.split('|');
    if (parts.length !== 4) throw new Error(`第 ${i + 1} 行应为 audio|speaker|language|text`);
    return { file: parts[0], speaker: parts[1], language: parts[2], text: parts[3] };
  });
  const selected = params.speaker ? rows.filter(r => r.speaker === params.speaker) : rows;
  if (!selected.length) throw new Error('没有匹配的角色台词；先确认角色映射。');
  const result = selected.map((r, i) => {
    const raw = r.voice_path || r.file;
    if (typeof raw !== 'string' || !raw) throw new Error(`第 ${i + 1} 条缺少音频路径`);
    const audio = isAbsolute(raw) ? raw : resolve(base, raw);
    if (!nonempty(audio)) throw new Error(`音频不存在或为空：${audio}`);
    if (typeof r.speaker !== 'string' || !r.speaker.trim() || typeof r.text !== 'string' || !r.text.trim()) throw new Error(`第 ${i + 1} 条缺角色或台词；不能当作已配对语料。`);
    return { ...r, voice: basename(audio), voice_path: audio.replace(/\\/g, '/'), scene: r.scene || `anime:${basename(base)}` };
  });
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, result.map(r => JSON.stringify(r)).join('\n') + '\n', { encoding: 'utf8', flag: 'wx' });
  return answer(true, `已桥接 ${result.length} 条原台词与音频路径：${out}\n下一步使用 gsv/scan、gsv/build，lines 参数传此文件。桥接不证明角色与台词正确，仍需人工校对。`,
    { stage: 'anime-bridge', lines: out, count: result.length, source: input });
}

export function createAnimeBackend(config = {}, runner = runOnce) {
  const c = { python: process.env.ANIME_PYTHON || 'python', ffmpeg: process.env.FFMPEG_BIN || 'ffmpeg', timeoutMs: 7_200_000, ...config };
  const env = {
    ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1', HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1',
    ANIME_EXTRACTOR_SCRIPTS: EXTRACTOR,
    ...(c.whisperModel ? { ANIME_WHISPER_MODEL: c.whisperModel } : {}),
    ...(c.camppModel ? { ANIME_CAMPP_MODEL: c.camppModel } : {}),
  };
  const invoke = async (script, args, expected = [], extra = {}) => {
    const r = await runner({ cmd: c.python, args: [script, ...args.map(String)], env, timeoutMs: c.timeoutMs });
    const artifacts = expected.map(path => ({ path, exists: nonempty(path) }));
    const ok = r.ok && artifacts.every(a => a.exists);
    return answer(ok, `${ok ? '完成' : '失败'}：番剧素材处理\n${(r.stdout || '').slice(-4000)}\n${(r.stderr || '').slice(-2000)}`,
      { stage: 'anime', exitCode: r.exitCode, artifacts, ...extra });
  };
  const protect = fn => async (params = {}) => {
    try { return await fn(params); } catch (e) { return answer(false, e.message, { stage: 'anime', error: e.message }); }
  };
  return new Map([
    ['extract', {
      name: 'anime_extract', description: '从本地番剧音视频提取候选说话人和台词；使用已下载的本地 ASR/声纹模型。角色编号需人工确认；含配乐时优先先分离或使用 rebuild。输出目录不覆盖。',
      parameters: schema({ input: s('本地音视频'), outDir: s('新的输出目录'), language: { type: 'string', enum: ['ja', 'zh', 'en', 'ko', 'yue'] }, numSpeakers: { type: 'integer' }, keep: s('保留片段，如 0-99,171-1324'), device: { type: 'string', enum: ['cuda', 'cpu'] } }, ['input', 'outDir']),
      execute: protect(async p => {
        if (!file(p.input)) throw new Error(`输入音视频不存在：${p.input}`);
        localModels(c);
        const out = freshDir(p.outDir);
        const args = ['run', resolve(p.input), '--out-dir', out, '--ffmpeg', c.ffmpeg, '--whisper-model', c.whisperModel, '--campp-model', c.camppModel];
        for (const [key, flag] of [['language', '--language'], ['numSpeakers', '--num-speakers'], ['keep', '--keep']]) if (p[key] !== undefined) args.push(flag, p[key]);
        if (p.device) args.push('--device', p.device, '--asr-device', p.device, '--asr-compute', p.device === 'cpu' ? 'int8' : 'int8_float16');
        return invoke(join(EXTRACTOR, 'anime_lines.py'), args, [join(out, 'lines.jsonl'), join(out, 'gpt_sovits.list')], { outDir: out, rolesDir: join(out, 'roles'), lines: join(out, 'lines.jsonl'), mapFile: join(out, 'speaker_map.json'), needsRoleReview: true });
      }),
    }],
    ['refine', {
      name: 'anime_refine', description: '使用人工确认的角色映射复核候选切片并导出目标角色。保留自然静默；输出必须是新目录，不覆盖已有结果。之后桥接台词再进入训练。',
      parameters: schema({ runDir: s('extract 输出目录，需保留 _work'), mapFile: s('人工确认的角色映射 JSON'), target: s('目标角色名'), outDir: s('新的输出目录'), language: { type: 'string', enum: ['ja', 'zh', 'en', 'ko', 'yue'] }, device: { type: 'string', enum: ['cuda', 'cpu'] } }, ['runDir', 'mapFile', 'target', 'outDir']),
      execute: protect(async p => {
        localModels(c, false);
        if (!file(p.mapFile)) throw new Error(`角色映射不存在：${p.mapFile}`);
        const out = freshDir(p.outDir);
        const args = ['refine', resolve(p.runDir), '--map', resolve(p.mapFile), '--target', p.target, '--out-dir', out, '--campp-model', c.camppModel, '--no-strip-silence'];
        if (p.language) args.push('--language', p.language);
        if (p.device) args.push('--device', p.device);
        return invoke(join(EXTRACTOR, 'anime_lines.py'), args, [join(out, 'lines.jsonl'), join(out, 'gpt_sovits.list')], { outDir: out, lines: join(out, 'lines.jsonl'), needsTextReview: true });
      }),
    }],
    ['rebuild', {
      name: 'anime_rebuild', description: '调用现有番剧整集重建流程：检查→抽音轨→分离→ASR→切片→目标声纹筛选→报告。止于训练集，不运行其内置训练/配音阶段。需要显式片段范围、人工参照和本地分离环境。',
      parameters: schema({ workDir: s('已有素材工作区，源文件布局 src/<episode>.m4a；输出固定为 _full/out/ts'), episodes: { type: 'array', minItems: 1, items: s('集数，如 ep01') }, role: s('角色名'), language: { type: 'string', enum: ['ja', 'zh', 'en'] }, keep: { type: 'array', minItems: 1, items: s('如 ep01=171-1324') }, ref: s('人工确认的角色参考音频目录') }, ['workDir', 'episodes', 'role', 'language', 'keep', 'ref']),
      execute: protect(async p => {
        localModels(c);
        if (!existsSync(p.ref) || !statSync(p.ref).isDirectory()) throw new Error(`角色参考音频目录不存在：${p.ref}`);
        if (!p.episodes.length || !p.episodes.every(e => /^[A-Za-z0-9_-]+$/.test(e))) throw new Error('集数不能为空且只能包含字母、数字、下划线、短横线。');
        for (const ep of p.episodes) if (!file(join(p.workDir, 'src', `${ep}.m4a`))) throw new Error(`缺少源文件 src/${ep}.m4a`);
        for (const key of ['sepRoot', 'sepPython', 'sepConfig', 'sepCheckpoint']) if (!c[key]) throw new Error(`缺少 pipelines.anime.${key}`);
        const out = freshDir(join(p.workDir, '_full', 'out', 'ts'));
        const args = ['--work', resolve(p.workDir), '--eps', p.episodes.join(','), '--role', p.role, '--lang', p.language, '--ref', resolve(p.ref), '--trainset', out,
          '--stages', 'probe,raw,sep,asr,slice,spk,report', '--python', c.python, '--ffmpeg', c.ffmpeg, '--extractor', EXTRACTOR,
          '--sep-root', c.sepRoot, '--sep-python', c.sepPython, '--sep-config', c.sepConfig, '--sep-checkpoint', c.sepCheckpoint];
        for (const keep of p.keep) args.push('--keep', keep);
        return invoke(REBUILDER, args, [join(out, 'train.list')], { trainset: out, listFile: join(out, 'train.list'), needsTextReview: true });
      }),
    }],
    ['bridge', { name: 'anime_bridge', description: '把番剧 lines.jsonl 或 train.list 桥接为 GPT-SoVITS 管线可读的 voice_path 台词清单。核对每条音频存在，保留角色、原文与来源；不覆盖文件。', parameters: schema({ input: s('lines.jsonl / train.list / gpt_sovits.list'), out: s('新的 JSONL 输出路径'), speaker: s('可选：精确选择一个已确认角色') }, ['input', 'out']), execute: protect(bridgeAnime) }],
  ]);
}

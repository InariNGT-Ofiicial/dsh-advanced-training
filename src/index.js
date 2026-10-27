/**
 * dsh-advanced-training: unified material, dataset, training and model-handoff entry.
 * DDSP/GPT-SoVITS/RVC/Applio/anime adapters retain their own formats and engines.
 * Original quality tools remain compatible; inference/cover/publishing are excluded
 * from the new training entry.
 */
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolveConfig } from './config.js';
import { runPy, clip } from './exec.js';
import { registerTrainingHub } from './training-hub.js';
import { loadTrainingConfig } from './training-config.js';

const PY_DIR = fileURLToPath(new URL('../py/', import.meta.url));

export const name = 'advanced-training';
export const inject = ['tools'];

const OUTPUT_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    rendered: { type: 'string' },
    stage: { type: 'string' },
  },
  required: ['ok', 'rendered'],
};

/** Cordis 的 output 形状是 { schema, render }，不是裸 schema */
const wrap = (run) => ({
  output: {
    schema: OUTPUT_SCHEMA,
    render: (_args, v) => [{ type: 'text', text: String(v?.rendered ?? '(无输出)') }],
  },
  execute: run,
});

export async function apply(ctx, cfg) {
  cfg = loadTrainingConfig(cfg || {});
  const { logger } = ctx;
  const tools = ctx.get('tools');
  const register = (def) => tools.register(def);
  const c = resolveConfig(cfg || {});
  const py = (script) => join(PY_DIR, script);

  // ── 1 · 三线体检 ─────────────────────────────────────────────────────
  register({
    name: 'adv_doctor',
    description:
      '三条模型线（DDSP / GPT-SoVITS / RVC）的工程体检：'
      + 'DDSP 整合包与科学计算 python（numpy/soundfile）、GSV 工程与 venv（faster-whisper）、'
      + 'rvc-voice-kit 工具链。本插件所有工具的前置检查。',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    ...wrap(async () => {
      const L = ['【先进训练插件 · 三线体检】'];
      const ok = {};

      // DDSP：出料用的科学计算 python
      L.push('', '── DDSP / 出料 ──');
      L.push(`  工程 ddspDir: ${c.ddspDir || '(未发现)'}`);
      L.push(`  python pyEnv: ${c.pyEnv}`);
      if (c.pyEnv && c.pyEnv !== 'python') {
        const r = await runPy(c.pyEnv, '-c',
          ['import json;import importlib.util as u;print(json.dumps({m: bool(u.find_spec(m)) for m in ("numpy","soundfile")}))'],
          60_000);
        const j = r.json || {};
        ok.py = !!(j.numpy && j.soundfile);
        L.push(`  numpy ${j.numpy ? '✅' : '❌'} · soundfile ${j.soundfile ? '✅' : '❌'}`
          + (ok.py ? '' : '  ← tail_audit 需要；缺了先补环境'));
      } else {
        ok.py = false;
        L.push('  ❌ 没有可用的科学计算 python（配 pyEnv 或装好 DDSP 整合包）');
      }

      // GSV：patch + ASR
      L.push('', '── GPT-SoVITS / 文本与 ASR ──');
      L.push(`  工程 gsvDir: ${c.gsvDir || '(未发现)'}`);
      ok.gsv = !!(c.gsvDir && existsSync(join(c.gsvDir, 'GPT_SoVITS', 'text', 'japanese.py')));
      L.push(`  文本链路: ${ok.gsv ? '✅ japanese.py/cleaner.py/symbols2.py 齐全' : '❌ 找不到 GPT_SoVITS/text/'}`);
      L.push(`  python gsvEnv: ${c.gsvEnv || '(未发现)'}`);
      if (c.gsvEnv) {
        const r = await runPy(c.gsvEnv, '-c',
          ['import json;import importlib.util as u;print(json.dumps({"fw": bool(u.find_spec("faster_whisper"))}))'],
          60_000);
        ok.fw = !!r.json?.fw;
        L.push(`  faster_whisper: ${ok.fw ? '✅' : '❌'}`);
      } else {
        ok.fw = false;
      }

      // RVC：工具链
      L.push('', '── RVC / 训练工具链 ──');
      L.push(`  工具链 rvcKitDir: ${c.rvcKitDir || '(未发现)'}`);
      ok.rvc = !!(c.rvcKitDir && existsSync(join(c.rvcKitDir, 'rvcx', 'voice.py')));
      L.push(`  rvcx.voice: ${ok.rvc ? '✅' : '❌'}`);
      L.push(`  上游引擎根: ${c.rvcRoot || c.rvcRootHint || '(未配置；请填写 pipelines.rvc)'}`);
      L.push('  本检查仅核对工具链；训练前调用 adv_train route=rvc action=doctor。');

      const all = ok.py && ok.gsv && ok.fw && ok.rvc;
      L.push('', all ? '✅ 三线齐备' : '⚠️ 有缺口（不影响可用部分的工具；缺口项会给出补法）');
      return { ok: all, stage: 'doctor', doctor: ok, rendered: L.join('\n') };
    }),
  });

  // ── 2 · 尾音体检 ─────────────────────────────────────────────────────
  register({
    name: 'adv_tail_audit',
    description:
      '对已有数据集/音频目录做「句尾体检」（RVC/DDSP/GSV 三线通用）：递归扫描 wav/flac，'
      + '输出 tail_ms / end_db / cut_ratio 与判读。判据与 dsh-train-pipeline 0.5.0 同口径：'
      + 'end_db > -12dB = 句子停在满电平；cut_ratio > 0.3 = 句尾被切断，训出来会吞末字。'
      + '★ 这是体检不是修复：数据已经有问题就重出料（对应管线插件开 tailPad），'
      + '短促句停在较高电平是正常的，看整体比例、别逐条判废。',
    parameters: {
      type: 'object',
      properties: {
        dir: { type: 'string', description: '音频目录（递归扫 wav/flac）' },
        json: { type: 'string', description: '报告写盘路径（可选）' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const dir = args.dir || '';
      if (!dir) {
        const msg = '❌ 缺 dir。给它要体检的数据集目录（如 <工作区>/<expName>/wav）。';
        return { ok: false, stage: 'tail-audit', rendered: msg };
      }
      if (!existsSync(dir)) {
        return { ok: false, stage: 'tail-audit', rendered: `❌ 目录不存在: ${dir}` };
      }
      const argv = ['--dir', dir];
      if (args.json) argv.push('--json', args.json);
      const r = await runPy(c.pyEnv, py('tail_audit.py'), argv, 600_000);
      const j = r.json;
      if (!j) return { ok: false, stage: 'tail-audit', rendered: `❌ 脚本无输出：\n${r.tail}` };
      if (!j.ok) return { ok: false, stage: 'tail-audit', detail: j, rendered: `❌ ${j.error || '体检失败'}` };

      const t = j.tail_stats;
      const L = [`【尾音体检】${j.dir}`];
      L.push(`音频 ${j.total} 个 · 测出 ${j.measured}${j.read_errors ? ` · 读失败 ${j.read_errors}` : ''}`);
      L.push(`收尾电平中位 ${t.median_end_db}dB · 尾静音中位 ${t.median_tail_ms}ms · 满电平收尾比例 ${t.cut_ratio}`);
      L.push(`判读：${t.cut_ratio > 0.3 ? '❌ ' : '✅ '}${t.verdict}`);
      if (t.cut_ratio > 0.3) {
        L.push('', '这类数据不要直接训：用对应管线插件重出料——'
          + 'DDSP: train_build（默认已留尾）；GSV: tts_build tailPad=true；RVC: 数据集同样需要留尾版切片。');
      }
      if (j.worst?.length) {
        L.push('', '最差样本（满电平收尾，前 20）：');
        for (const w of j.worst.slice(0, 20)) L.push(`  ${w.file}  ${w.end_db}dB / ${w.tail_ms}ms`);
      }
      if (j.report) L.push('', `报告: ${j.report}`);
      return { ok: t.cut_ratio <= 0.3, stage: 'tail-audit', audit: j, rendered: L.join('\n') };
    }),
  });

  // ── 3 · 消 UNK 补丁（GSV 专用）───────────────────────────────────────
  register({
    name: 'adv_unk_patch',
    description:
      '给 GPT-SoVITS 工程打「消 UNK」文本补丁。★ 仅适用 GSV：文本符号链路只有它有；'
      + 'DDSP/RVC 是 SVC 不吃文本，无 UNK 问题。'
      + 'japanese.py 剥无音值符号（「」『』～―（）()），cleaner.py 过滤声调短语边界 "#"——'
      + '两类字符都不在 symbols2 的 732 项里，会被降级成 UNK（所有词边界共享同一个 embedding）。'
      + '★ 幂等、自动备份 .bak_dsh_unk、可 revert、打前核对符号表。默认只报计划，apply=true 才写盘。'
      + '★ 打了必须重训：实测只打补丁不重训是负收益（PER 0.197→0.236）。'
      + '旧权重做对照推理时设 GSV_LEGACY_UNK=1 还原上游行为。',
    parameters: {
      type: 'object',
      properties: {
        apply: { type: 'boolean', description: '真的写盘（默认只报计划）' },
        revert: { type: 'boolean', description: '从 .bak_dsh_unk 回滚' },
        gsvRoot: {
          type: 'string',
          description: '显式指定 GPT-SoVITS 根目录。不传则用配置/自动发现——'
            + '写盘前请核对渲染里报出的实际路径',
        },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const root = args.gsvRoot || c.gsvDir;
      if (!root) {
        return { ok: false, stage: 'unk-patch', rendered: '❌ 未发现 GPT-SoVITS 工程；先跑 adv_doctor 或传 gsvRoot' };
      }
      const argv = ['--gsv-root', root];
      if (args.apply) argv.push('--apply');
      if (args.revert) argv.push('--revert');
      const r = await runPy(c.pyEnv, py('patch_unk.py'), argv, 180_000);
      const j = r.json;
      if (!j) return { ok: false, stage: 'unk-patch', rendered: `❌ 补丁脚本无输出：\n${r.tail}` };

      const L = [`【消 UNK 补丁】${j.gsv_root || root}`];
      if (!j.ok) {
        L.push(`❌ ${j.error || '失败'}`);
        for (const e of j.errors || []) L.push(`  ${e.file}: ${e.error}`);
        for (const f of j.files || []) if (f.error) L.push(`  ${f.file}: ${f.error}`);
        return { ok: false, stage: 'unk-patch', detail: j, rendered: L.join('\n') };
      }
      for (const f of j.files || []) L.push(`  ${f.file}: ${f.action} — ${f.note || ''}`);
      if (j.symbols?.checked) {
        L.push(`符号表 ${j.symbols.total} 项${j.symbols.conflicts?.length ? `，冲突: ${j.symbols.conflicts.join(' ')}` : '，无冲突'}`);
      }
      L.push('', j.applied ? '已写盘。下一步：重训（重算文本特征），旧权重对照用 GSV_LEGACY_UNK=1。'
        : j.reverted ? '已回滚。' : '这是计划；apply=true 才写盘。');
      return { ok: true, stage: 'unk-patch', detail: j, rendered: L.join('\n') };
    }),
  });

  // ── 4 · 验收 A/B ─────────────────────────────────────────────────────
  register({
    name: 'adv_ab_eval',
    description:
      '验收 A/B（无参考度量，三线通用）：对已合成的 wav 目录跑 ASR 回转写，'
      + '产出 CER（字符错误率）、末尾字覆盖率（tail_hit，默认末三字——切尾模型会吞末字，'
      + '实测 33%→93% 的直接验收指标）与尾音统计。'
      + '★ 本工具不做合成：先把 A/B 各自合成好（GSV→tts_infer；DDSP→train_infer；'
      + 'RVC→python -m rvcx.voice convert），再拿两个产物目录来比。'
      + '★ ASR 也会犯错：看 A vs B 的**相对差**，别拿绝对 CER 单独下结论。'
      + '★ 用 GSV venv 的 python（faster-whisper 在那里），配 gsvEnv 或 adv_doctor 先体检。',
    parameters: {
      type: 'object',
      properties: {
        aDir: { type: 'string', description: 'A 侧 wav 目录' },
        bDir: { type: 'string', description: 'B 侧 wav 目录（给了就出 A/B 对比）' },
        refsFile: { type: 'string', description: '参考文本 json：{文件名或词干: 台词}' },
        k: { type: 'number', description: '末尾字覆盖的字符数（默认 3）' },
        lang: { type: 'string', description: 'auto/zh/ja/en（默认 auto）' },
        model: { type: 'string', description: 'whisper 模型（默认 large-v3）' },
        limit: { type: 'number', description: '每侧最多评多少条（0=全量）' },
        allowDownload: { type: 'boolean', description: '本地无 whisper 缓存时允许联网下载（默认离线）' },
        json: { type: 'string', description: '报告写盘路径' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      if (!args.aDir || !existsSync(args.aDir)) {
        return { ok: false, stage: 'ab-eval', rendered: `❌ aDir 缺失或不存在: ${args.aDir || '(空)'}` };
      }
      if (!c.gsvEnv) {
        return { ok: false, stage: 'ab-eval',
          rendered: '❌ 没有 GSV venv 的 python（faster-whisper 在那里）。配 gsvEnv 后重试。' };
      }
      const argv = ['--dir', args.aDir, '--model', args.model || c.asrModel,
        '--compute-type', c.asrComputeType, '--device', 'cuda',
        '--lang', args.lang || 'auto', '--k', String(args.k ?? 3)];
      if (args.bDir) argv.push('--dir-b', args.bDir);
      if (args.refsFile) argv.push('--refs', args.refsFile);
      if (args.limit) argv.push('--limit', String(args.limit));
      if (c.asrCacheDir) argv.push('--cache-dir', c.asrCacheDir);
      if (args.allowDownload) argv.push('--allow-download');
      if (args.json) argv.push('--json', args.json);

      const r = await runPy(c.gsvEnv, py('asr_cer.py'), argv, 7_200_000);
      const j = r.json;
      if (!j) return { ok: false, stage: 'ab-eval', rendered: `❌ 脚本无输出：\n${r.tail}` };
      if (!j.ok) return { ok: false, stage: 'ab-eval', detail: j, rendered: `❌ ${j.error || '评测失败'}` };

      const side = (tag, s) => {
        const L = [`【${tag}】${s.dir}`];
        L.push(`  ${s.n} 条 · 有参考 ${s.with_ref} · CER 均值 ${s.cer_mean ?? '—'} / 中位 ${s.cer_median ?? '—'}`
          + ` · 末${j.k}字命中 ${(s.tail_hit_rate ?? 0) * 100}%`);
        const t = s.tail_stats;
        if (t) L.push(`  尾音: 中位收尾 ${t.median_end_db}dB · 尾静音 ${t.median_tail_ms}ms · 满电平比例 ${t.cut_ratio}`);
        return L;
      };
      const L = [`【验收 A/B】model=${j.model}`];
      L.push(...side('A', j.a));
      if (j.b) {
        L.push('', ...side('B', j.b));
        const m = j.compare;
        L.push('', '对比（A → B）：');
        L.push(`  CER 均值   ${m.cer_mean[0] ?? '—'} → ${m.cer_mean[1] ?? '—'}`);
        L.push(`  末${j.k}字命中 ${m.tail_hit_rate[0] ?? '—'} → ${m.tail_hit_rate[1] ?? '—'}`);
        L.push(`  收尾电平   ${m.median_end_db[0] ?? '—'}dB → ${m.median_end_db[1] ?? '—'}dB`
          + ` · 满电平比例 ${m.cut_ratio[0]} → ${m.cut_ratio[1]}`);
      }
      if (j.a.worst_cer?.length) {
        L.push('', 'A 侧 CER 最差（前 10，人工抽查用）：');
        for (const w of j.a.worst_cer) L.push(`  ${w.file}  cer=${w.cer}  ref="${clip(w.ref, 40)}" asr="${clip(w.asr, 40)}"`);
      }
      if (j.report) L.push('', `报告: ${j.report}`);
      L.push('', '提醒：CER 是第三方 ASR 给的相对证据，末尾字命中与尾音统计才是结构性信号；两者一起看。');
      return { ok: true, stage: 'ab-eval', eval: j, rendered: L.join('\n') };
    }),
  });

  await registerTrainingHub(ctx, cfg || {});

  logger.info(`[advanced-training] 已注册 7 个工具（含统一训练入口） · ddsp=${c.ddspDir || '(无)'} · gsv=${c.gsvDir || '(无)'} · rvc=${c.rvcKitDir || '(无)'}`);
}

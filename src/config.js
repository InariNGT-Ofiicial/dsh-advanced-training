/**
 * config.js —— 三条模型线的工程发现（DDSP / GPT-SoVITS / RVC）。
 *
 * 原则与 dsh-train-pipeline / dsh-tts-train-pipeline 一致：
 *   环境变量 > 显式配置 > 自动发现。发现的只是一层「默认值」，
 *   任何写盘类工具都必须在渲染里报出**实际使用的路径**。
 *
 * ★ 教训（2026-10-25 实录）：「路径不像 X 就回退自动发现」在写入类操作里
 *   是定时炸弹——曾把临时目录静默换成真实工程。所以本插件对写盘类工具
 *   （unk_patch）一律以渲染路径为准，且 patch 脚本自身有 already-external 保护。
 */
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const env = (k) => process.env[k] || '';

/** 在 dir 下找名字以 prefix 开头、且存在 marker 的子目录 */
function findSubdir(dir, prefix, marker) {
  if (!dir || !existsSync(dir)) return '';
  for (const name of readdirSync(dir)) {
    if (!name.startsWith(prefix)) continue;
    const p = join(dir, name);
    if (!marker || existsSync(join(p, marker))) return p;
  }
  return '';
}

export function resolveConfig(cfg = {}) {
  const c = { ...cfg };

  // ── 根目录（环境变量 > 配置 > 本机默认）────────────────────────────
  const songRoot = c.songRoot || env('DSH_COVER_ROOT') || 'D:/AI-Song';
  c.songRoot = songRoot;

  // ── DDSP 线（科学计算 python 在它的整合包 env 里）──────────────────
  c.ddspDir = c.ddspDir || env('DSH_DDSP_DIR') || findSubdir(songRoot, '02_', 'train_reflow.py') || '';
  if (!c.pyEnv) {
    const cand = [
      c.pyEnv,
      env('DSH_TRAIN_PYTHON'),
      c.rvcEnv || c.pipelines?.rvc?.python,
      c.applioEnv || c.pipelines?.applio?.python,
      c.ddspDir ? join(c.ddspDir, 'env/python.exe') : '',
      'D:/AI-Song/01_人声分离_MSST-GUI/env/python.exe',
    ].filter(Boolean);
    c.pyEnv = cand.find((p) => existsSync(p)) || 'python';
  }

  // ── GPT-SoVITS 线（faster-whisper 在它的 venv 里）─────────────────
  c.gsvDir = c.gsvDir || env('DSH_GSV_ROOT')
    || (existsSync('D:/AI-Voice/02_GPT-SoVITS') ? 'D:/AI-Voice/02_GPT-SoVITS' : '');
  if (!c.gsvEnv) {
    const cand = [
      env('DSH_GSV_PYTHON'),
      c.gsvDir ? join(c.gsvDir, '.venv/Scripts/python.exe') : '',
    ].filter(Boolean);
    c.gsvEnv = cand.find((p) => existsSync(p)) || '';
  }
  c.asrModel = c.asrModel || 'large-v3';
  c.asrComputeType = c.asrComputeType || 'float16';
  if (!c.asrCacheDir) {
    // faster-whisper 的本机缓存：GSV 工程同级的 hf_cache（2026-10-26 实测在
    // D:/AI-Voice/hf_cache/hub/models--Systran--faster-whisper-large-v3）。
    const cand = [env('DSH_ASR_CACHE'), c.gsvDir ? join(c.gsvDir, '..', 'hf_cache') : '']
      .filter(Boolean);
    c.asrCacheDir = cand.find((p) => existsSync(p)) || '';
  }

  // ── RVC 线（rvc-voice-kit 工具链 + 上游引擎根）─────────────────────
  c.rvcKitDir = c.rvcKitDir || env('DSH_RVC_KIT') || fileURLToPath(new URL('../vendor/rvc-voice-kit/', import.meta.url));
  c.rvcRoot = c.rvcRoot || c.pipelines?.rvc?.root || env('DSH_RVC_ROOT') || '';   // RVC 安装根（setup.ps1 -Root）
  if (c.rvcKitDir && !c.rvcRoot) {
    // rvcx.local.toml 里记了上游根；只做「有没有」级别的提示，不解析 toml
    c.rvcRootHint = existsSync(join(c.rvcKitDir, 'rvcx.local.toml')) ? 'rvcx.local.toml' : '';
  }

  // ── 运行控制 ────────────────────────────────────────────────────────
  c.maxOutputChars = Number(c.maxOutputChars) || 6000;

  return c;
}

import { existsSync, statSync } from 'node:fs';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRvcBackend } from './rvc-backend.js';

export function resolveApplioConfig(raw = {}) {
  const repo = raw.repo || raw.root || raw.applioRoot || process.env.DSH_APPLIO_ROOT || '';
  const workRoot = resolve(raw.workRoot || process.cwd(), 'applio');
  const sampleRate = raw.sampleRate || '48k';
  const vocoder = raw.vocoder || 'HiFi-GAN';
  const pretrained = repo ? join(repo, 'rvc/models/pretraineds', vocoder === 'RefineGAN' ? 'refinegan' : 'hifi-gan') : '';
  return { root: repo, repo, workRoot, logsDir: raw.logsDir ? resolve(raw.logsDir) : repo ? resolve(repo, 'logs') : '',
    python: raw.python || raw.applioEnv || process.env.DSH_APPLIO_PYTHON || (repo ? join(repo, 'env', process.platform === 'win32' ? 'python.exe' : 'bin/python') : ''),
    ffmpegDir: raw.ffmpegDir || (raw.ffmpeg && existsSync(raw.ffmpeg) ? dirname(resolve(raw.ffmpeg)) : ''),
    expName: raw.expName || 'my-applio-voice', modelDir: resolve(raw.modelDir || join(workRoot, 'models')),
    sampleRate, vocoder, precision: raw.precision || 'fp16', f0Method: raw.f0Method || 'rmvpe',
    embedderDir: raw.embedderDir || (repo ? join(repo, 'rvc/models/embedders/contentvec') : ''),
    pretrainedG: raw.pretrainedG || (pretrained ? join(pretrained, `f0G${sampleRate}.pth`) : ''),
    pretrainedD: raw.pretrainedD || (pretrained ? join(pretrained, `f0D${sampleRate}.pth`) : ''),
    epochs: raw.epochs ?? 150, batchSize: raw.batchSize ?? 4, saveEvery: raw.saveEvery ?? 10,
    processes: raw.processes ?? 2, minSlices: raw.minSlices ?? 1,
    cutMode: raw.cutMode || 'Automatic', normalization: raw.normalization || 'none' };
}

export function createApplioBackend(raw = {}, deps = {}) {
  return createRvcBackend(raw, deps, { route: 'applio', label: 'Applio',
    script: fileURLToPath(new URL('../py/applio_route.py', import.meta.url)), resolveConfig: resolveApplioConfig,
    sampleRates: ['24k','32k','40k','48k'], adopt: true,
    doctorProperties: { resumeOnly: { type:'boolean', description:'仅检查已有实验续训的工程与依赖；底模和重新提取资源留给新训练检查。' } },
    check(c) {
      if (!c.repo || !existsSync(c.repo) || !statSync(c.repo).isDirectory() || !c.python || !existsSync(c.python) || !statSync(c.python).isFile()) throw new Error('请配置 pipelines.applio 的 repo/root 和已有训练环境 python；先调用 applio/doctor。');
      if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(c.expName)) throw new Error('expName 必须是安全 ASCII 实验名。');
      if (!(c.vocoder === 'HiFi-GAN' ? ['32k','40k','48k'] : c.vocoder === 'RefineGAN' ? ['24k','32k'] : []).includes(c.sampleRate)) throw new Error('HiFi-GAN 支持 32k/40k/48k；RefineGAN 支持 24k/32k。');
      if (!['rmvpe','crepe','fcpe','swift'].includes(c.f0Method) || !['fp16','fp32','bf16'].includes(c.precision) || !['Automatic','Skip'].includes(c.cutMode) || !['none','pre','post'].includes(c.normalization)) throw new Error('Applio F0、精度、切片或归一化配置无效。');
      for (const k of ['epochs','batchSize','saveEvery','processes','minSlices']) if (!Number.isInteger(c[k]) || c[k] < 1) throw new Error(`${k} 必须是正整数。`);
    },
    descriptions: {
      doctor: '核对 Applio 训练接口版本、Python/CUDA、实际 logsDir、预训练权重、离线 embedder、F0 模型和静音样本。',
      build: '后台调用 Applio 原生预处理到 sliced_audios；保留源目录并验证音频规格。已有出料仅显式 resume 且指纹一致时复用。',
      preprocess: '后台调用 Applio 原生 F0 与特征提取，验证 extracted/f0/f0_voiced、配置和完整单角色清单。',
      run: '在插件内直接启动 Applio 训练，固定 cleanup=false。已有权重拒绝重开；继续训练用 resume 操作，自动独立备份并严格检查 G/D、优化器和 AMP 恢复。',
      index: '后台调用 Applio 原生索引构建并读取验证成品 <expName>.index 的维度与向量数。',
      register: '核验 Applio 导出音色模型、索引、配置、清单、模型信息与日志，记录哈希并交接；拒绝覆盖，文件通过不代表听感验收。',
    } });
}

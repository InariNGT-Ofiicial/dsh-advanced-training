/** Optional integration against real Python/FFmpeg, using synthetic audio only. */
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve, basename } from 'node:path';
import { execFileSync } from 'node:child_process';
import { createTrainingHub } from '../src/training-hub.js';
import { bridgeAnime } from '../src/anime-backend.js';

const python = process.env.TEST_AUDIO_PYTHON;
const ffmpeg = process.env.TEST_FFMPEG;
assert(python && ffmpeg, 'Set TEST_AUDIO_PYTHON (numpy/soundfile) and TEST_FFMPEG');
const root = mkdtempSync(join(tmpdir(), 'dsh-training-local-'));
const probe = join(dirname(ffmpeg), process.platform === 'win32' ? 'ffprobe.exe' : 'ffprobe');
function writeWave(path, hz) {
  const sr = 44100; const n = sr * 3; const b = Buffer.alloc(44 + n * 2);
  b.write('RIFF'); b.writeUInt32LE(b.length - 8, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16);
  b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    const t = i / sr; const envelope = t < .15 ? 0 : t > 2.7 ? 0 : t > 2.2 ? Math.exp(-(t - 2.2) * 13) : .8;
    b.writeInt16LE(Math.round(Math.sin(2 * Math.PI * hz * t) * envelope * 14000), 44 + i * 2);
  }
  writeFileSync(path, b);
}
try {
  const corpus = join(root, 'corpus'); const voiceA = join(corpus, 'voice', '角色A'); const voiceB = join(corpus, 'voice', '角色B');
  mkdirSync(voiceA, { recursive: true }); mkdirSync(voiceB, { recursive: true });
  writeWave(join(voiceA, 'a1.wav'), 220); writeWave(join(voiceA, 'a2.wav'), 260); writeWave(join(voiceB, 'b1.wav'), 170);
  const ddsp = join(root, 'ddsp-engine'); const gsv = join(root, 'gsv-engine');
  mkdirSync(ddsp); mkdirSync(join(gsv, 'GPT_SoVITS'), { recursive: true }); writeFileSync(join(gsv, 'webui.py'), '# fixture');
  const hub = await createTrainingHub({ logger: { info() {}, warn() {}, error() {} }, get() {} }, {
    songRoot: root, workRoot: join(root, 'work'), ddspDir: ddsp, ddspEnv: python, gsvDir: gsv, gsvEnv: python, ffmpeg,
    pipelines: { ddsp: { trainRoot: join(root, 'ddsp-work'), expName: 'fixture', minSec: .5, maxSec: 5 }, gsv: { version: 'v2', expName: 'fixture', minSec: .5, maxSec: 5 } },
  });
  console.log('DDSP scan');
  const ddspScan = await hub.execute({ route: 'ddsp', action: 'scan', params: { input: corpus } });
  assert(ddspScan.ok, ddspScan.rendered);
  console.log('DDSP build (native sample rate; no optional resampler initialization)');
  const ddspBuild = await hub.execute({ route: 'ddsp', action: 'build', params: { input: corpus, speakers: ['角色A'], minSec: .5, maxSec: 5 } });
  assert(ddspBuild.ok, ddspBuild.rendered);
  assert(ddspBuild.datasets.length > 0);
  const ds = ddspBuild.datasets[0];
  assert.equal(ds.built, 2);
  const lines = join(root, 'anime-lines.jsonl');
  writeFileSync(lines, [1, 2].map(i => JSON.stringify({ speaker: '角色A', text: `合成测试语料${i}。`, file: join(voiceA, `a${i}.wav`) })).join('\n') + '\n');
  const bridge = bridgeAnime({ input: lines, out: join(root, 'gsv-lines.jsonl'), speaker: '角色A' });
  console.log('GPT-SoVITS scan');
  const gsvScan = await hub.execute({ route: 'gsv', action: 'scan', params: { source: bridge.lines, minSec: .5, maxSec: 5 } });
  assert(gsvScan.ok, gsvScan.rendered);
  console.log('GPT-SoVITS build');
  const gsvBuild = await hub.execute({ route: 'gsv', action: 'build', params: { lines: bridge.lines, speaker: '角色A', lang: 'zh', minSec: .5, maxSec: 5 } });
  assert(gsvBuild.ok, gsvBuild.rendered);
  assert.equal(gsvBuild.build.lines, 2);
  const list = readFileSync(gsvBuild.build.list, 'utf8');
  assert(list.includes('合成测试语料1。') && list.includes('合成测试语料2。'));
  assert(existsSync(ddspBuild.receipt) && existsSync(gsvBuild.receipt));
  const wavName = list.trim().split(/\r?\n/)[0].split('|')[0];
  const wavPath = resolve(gsvBuild.build.wav_dir, wavName);
  const audioInfo = JSON.parse(execFileSync(probe, ['-v', 'error', '-show_streams', '-of', 'json', wavPath], { encoding: 'utf8' })).streams[0];
  assert.equal(Number(audioInfo.sample_rate), 32000); assert.equal(audioInfo.channels, 1);
  console.log(JSON.stringify({ passed: true, input: 'synthetic PCM WAV with natural decay', ddsp: { scan: ddspScan.ok, built: ds.built }, gsv: { scan: gsvScan.ok, built: gsvBuild.build.lines, sampleRate: Number(audioInfo.sample_rate), channels: audioInfo.channels, originalTextPreserved: true }, animeTextBridge: true, receipts: true, gpuTrainingPerformed: false, animeModelInferencePerformed: false }, null, 2));
} finally {
  assert.equal(dirname(resolve(root)), resolve(tmpdir())); assert(basename(root).startsWith('dsh-training-local-'));
  rmSync(root, { recursive: true, force: true });
}

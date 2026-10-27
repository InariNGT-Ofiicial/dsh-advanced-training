/** Explicit local RVC test: synthetic audio, unique experiment, real training. */
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync, readFileSync, createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { createTrainingHub } from '../src/training-hub.js';

const root = process.env.TEST_RVC_ROOT;
const python = process.env.TEST_RVC_PYTHON;
const work = process.env.TEST_RVC_WORK;
assert(root && python && work, 'Set TEST_RVC_ROOT, TEST_RVC_PYTHON and TEST_RVC_WORK; this test starts real RVC training.');
const exp = 'dsh_smoke_'+Date.now();
const input = resolve(work, 'synthetic-'+exp);
mkdirSync(input, { recursive: true });
function wave(path, hz) {
  const sr = 48000, n = sr * 4, b = Buffer.alloc(44 + n * 2);
  b.write('RIFF'); b.writeUInt32LE(b.length - 8, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16);
  b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  let seed = hz;
  for (let i = 0; i < n; i++) {
    const t = i / sr; const envelope = Math.min(1, t/.15, Math.max(0, (3.8 - t)/.3));
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const noise = (seed / 4294967296 - .5) * .015;
    const phase = 2 * Math.PI * (hz * t + 2 * t * t);
    const sample = (.5 * Math.sin(phase) + .2 * Math.sin(2 * phase) + .1 * Math.sin(3 * phase) + noise) * envelope;
    b.writeInt16LE(Math.round(sample * 16000), 44 + i * 2);
  }
  writeFileSync(path, b);
}
for (let i = 0; i < 4; i++) wave(join(input, `tone-${i}.wav`), 170 + i * 35);
const hub = await createTrainingHub({ logger: { info() {}, warn() {}, error() {} }, get() {} }, {
  workRoot: work,
  pipelines: { rvc: { root, python, workRoot: work, ffmpegDir: process.env.TEST_RVC_FFMPEG_DIR || '', expName: exp, sampleRate: '48k', f0Method: 'pm', epochs: 1, batchSize: 1, saveEvery: 1, minSlices: 1, processes: 2 } },
});
const report = { checked_date: new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Shanghai' }).format(new Date()), experiment: exp, input: 'four synthetic 48kHz mono harmonic/noise WAV files', inferencePerformed: false, subjectiveQualityAccepted: false, stages: [] };
const save = () => {
  if (process.env.TEST_RVC_REPORT) {
    mkdirSync(join(resolve(process.env.TEST_RVC_REPORT), '..'), { recursive: true });
    writeFileSync(process.env.TEST_RVC_REPORT, JSON.stringify(report, null, 2));
  }
};
async function call(action, params = {}) {
  return hub.execute({ route: 'rvc', action, params: { ...params, ...(action === 'scan' ? {} : { expName: params.expName || exp }) } });
}
async function wait(start, name, acceptFailure = false) {
  assert(start.ok, start.rendered);
  const deadline = Date.now() + 15 * 60 * 1000;
  while (Date.now() < deadline) {
    const r = await call('status', { expName: start.expName, jobId: start.jobId });
    if (!['queued', 'running'].includes(r.state)) {
      if (!acceptFailure) assert(r.ok && r.state === 'succeeded', r.rendered);
      report.stages.push({ name, state: r.state, jobId: r.jobId, artifacts: r.artifacts, error: r.error, logPath: r.logPath, backupDir: r.backupDir, resume: r.resume });
      save(); console.log(`${name}: ${r.state}`);
      return r;
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
  throw new Error(`Timed out waiting for ${name}; inspect ${start.logPath}, job ${start.jobId}`);
}
try {
  let r = await call('doctor', { deep: true }); assert(r.ok, r.rendered); console.log('RVC doctor passed');
  r = await call('scan', { input }); assert(r.ok && r.count === 4, r.rendered); console.log('RVC scan passed');
  await wait(await call('build', { input }), 'build');
  await wait(await call('preprocess'), 'preprocess');
  const missing = await wait(await call('run', { epochs: 3, resume: true }), 'reject-resume-without-gd', true);
  assert(!missing.ok && missing.error.includes('续训缺少 G_'));
  await wait(await call('run', { epochs: 2, batchSize: 1, saveEvery: 1 }), 'train-two-epochs');
  const refused = await wait(await call('run', { epochs: 3 }), 'reject-implicit-overwrite', true);
  assert(!refused.ok && refused.state === 'failed' && refused.error.includes('resume=true'));
  const resumed = await wait(await call('run', { epochs: 3, batchSize: 1, saveEvery: 1, resume: true }), 'resume-from-two-to-three');
  assert(resumed.resume.confirmed && resumed.resume.fromEpoch === 2 && resumed.resume.targetEpoch === 3);
  assert(resumed.rendered.includes('恢复已确认'));
  const backup = JSON.parse(readFileSync(join(resumed.backupDir, 'backup.json'), 'utf8'));
  assert(backup.complete && backup.fromEpoch === 2 && backup.files.length === 4);
  for (const entry of backup.files) {
    const hash = createHash('sha256');
    for await (const chunk of createReadStream(join(resumed.backupDir, entry.file))) hash.update(chunk);
    assert.equal(hash.digest('hex'), entry.sha256);
  }
  report.resumeBackupVerified = true;
  await wait(await call('index'), 'index');
  r = await call('register', { role: '合成语料流程测试' }); assert(r.ok, r.rendered);
  const manifest = JSON.parse(readFileSync(r.manifest, 'utf8'));
  assert.equal(manifest.route, 'rvc'); assert(manifest.index.vectors > 0); assert.equal(manifest.model.epochs, 3);
  assert.equal(manifest.qualityAccepted, false);
  report.register = { outDir: r.outDir, manifest: r.manifest, artifacts: r.artifacts, epoch: manifest.model.epochs, indexVectors: manifest.index.vectors };
  const overwrite = await call('register', { outDir: r.outDir }); assert(!overwrite.ok && overwrite.rendered.includes('不覆盖'));
  report.rejectBundleOverwrite = true;
  const cancellation = await call('build', { expName: exp+'_stop', input });
  assert(cancellation.ok);
  for (let i = 0; i < 40; i++) {
    const s = await call('status', { expName: cancellation.expName, jobId: cancellation.jobId });
    if (s.heartbeatAt) break;
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  const stop = await call('stop', { expName: cancellation.expName, jobId: cancellation.jobId }); assert(stop.ok, stop.rendered);
  const stopped = await wait(cancellation, 'owned-job-cancel', true); assert.equal(stopped.state, 'stopped');
  report.passed = true; save(); console.log(JSON.stringify(report, null, 2));
} catch (e) {
  report.passed = false; report.error = String(e.message); save(); throw e;
}

import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve, basename } from 'node:path';
import { createRvcBackend } from '../src/rvc-backend.js';
import { createTrainingHub } from '../src/training-hub.js';
import { loadTrainingConfig } from '../src/training-config.js';

const root = mkdtempSync(join(tmpdir(), 'dsh-rvc-integration-'));
try {
  let calls = 0;
  let launched;
  const cfg = { root, repo: root, python: process.execPath, workRoot: root, expName: 'fixture', ffmpegDir: root };
  const backend = createRvcBackend(cfg, {
    async runOnce(r) {
      calls += 1;
      assert.equal(r.cmd, process.execPath);
      assert(r.env.PATH.startsWith(root));
      const spec = JSON.parse(r.args[r.args.indexOf('--request') + 1]);
      assert.equal(spec.action, 'scan');
      assert.equal(spec.params.input, 'path with spaces');
      return { ok: true, stdout: JSON.stringify({ ok: true, count: 2, seconds: 5 }), stderr: '', exitCode: 0 };
    },
    async spawnJob(r) {
      launched = JSON.parse(readFileSync(r.args[r.args.indexOf('--job') + 1], 'utf8'));
      assert.equal(launched.action, 'run');
      assert.equal(launched.params.expName, 'fixture');
      assert(!JSON.stringify(launched).includes('"steps"'));
      writeFileSync(launched.statePath, JSON.stringify({ ok: true, state: 'running', pid: process.pid, jobId: launched.jobId, heartbeatAt: Date.now()/1000 }));
      return { pidValid: true, pid: process.pid };
    },
  });
  const scan = await backend.get('scan').execute({ input: 'path with spaces' });
  assert(scan.ok && scan.count === 2 && calls === 1);
  const run = await backend.get('run').execute({ expName: 'fixture', epochs: 1, resume: true });
  assert(run.ok && run.state === 'started', JSON.stringify(run));
  let status = await backend.get('status').execute({ jobId: run.jobId });
  assert(status.ok && status.state === 'running');
  const stop = await backend.get('stop').execute({ jobId: run.jobId });
  assert(stop.ok && existsSync(launched.stopPath));
  assert.equal(readFileSync(launched.stopPath, 'utf8'), launched.jobId);
  assert(!(await backend.get('stop').execute({ jobId: '../unowned' })).ok);
  writeFileSync(launched.statePath, JSON.stringify({ ok: false, state: 'failed', error: 'fixture engine failure', jobId: launched.jobId }));
  status = await backend.get('status').execute({ jobId: run.jobId });
  assert(!status.ok && status.state === 'failed' && status.rendered.includes('fixture engine failure'));
  writeFileSync(launched.statePath, JSON.stringify({ ok: true, state: 'running', pid: process.pid, jobId: launched.jobId, heartbeatAt: 1 }));
  status = await backend.get('status').execute({ jobId: run.jobId });
  assert(!status.ok && status.state === 'unknown', 'stale state cannot override the calculated failure');
  writeFileSync(launched.statePath, JSON.stringify({ ok: true, state: 'running', pid: 2147483647, jobId: launched.jobId, heartbeatAt: Date.now()/1000 }));
  status = await backend.get('status').execute({ jobId: run.jobId });
  assert(status.ok && status.state === 'running', 'fresh heartbeat tolerates a transient process-probe failure');
  writeFileSync(launched.statePath, JSON.stringify({ ok: true, state: 'running', pid: 2147483647, jobId: launched.jobId, heartbeatAt: 1 }));
  status = await backend.get('status').execute({ jobId: run.jobId });
  assert(!status.ok && status.state === 'lost', 'a dead process cannot remain successful');
  writeFileSync(launched.statePath, JSON.stringify({ ok: true, state: 'running', pid: process.pid, jobId: launched.jobId, heartbeatAt: Date.now()/1000,
    backupDir: 'independent-backup', resume: { fromEpoch: 10, targetEpoch: 11, confirmed: true } }));
  status = await backend.get('status').execute({ jobId: run.jobId });
  assert(status.ok && status.rendered.includes('第 10 轮') && status.rendered.includes('恢复已确认') && status.rendered.includes('independent-backup'));
  const configFile = join(root, 'local.json');
  writeFileSync(configFile, JSON.stringify({ pipelines: { rvc: { expName: 'from-file' } } }));
  const merged = loadTrainingConfig({ trainingConfig: configFile, pipelines: { rvc: { root, expName: 'old' } } });
  assert.equal(merged.pipelines.rvc.root, root); assert.equal(merged.pipelines.rvc.expName, 'from-file');
  const hub = await createTrainingHub({ logger: { info() {} }, get() {} }, { workRoot: root });
  const cat = hub.catalog({ route: 'rvc' });
  assert.deepEqual(cat.routes[0].actions.map(a => a.action), ['doctor', 'scan', 'build', 'preprocess', 'run', 'index', 'status', 'stop', 'register']);
  for (const action of ['infer', 'convert', 'all']) assert(!(await hub.execute({ route: 'rvc', action })).ok);
  for (const request of [
    { action: 'run', params: { expName: '../escape' } },
    { action: 'run', params: { epochs: 0 } },
    { action: 'run', params: { steps: 'train,index,infer' } },
    { action: 'build', params: { input: root, sampleRate: '32k' } },
  ]) {
    const r = await hub.execute({ route: 'rvc', ...request, dryRun: true });
    assert(!r.ok && !r.executed);
  }
  console.log('PASS RVC: route registration, structured invocation, owned jobs/stop requests, error/stale-state propagation, config merge and training-only parameter boundaries. No real training in this test.');
} finally {
  assert.equal(dirname(resolve(root)), resolve(tmpdir())); assert(basename(root).startsWith('dsh-rvc-integration-'));
  rmSync(root, { recursive: true, force: true });
}

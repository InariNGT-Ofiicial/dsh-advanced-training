import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, readdirSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname, basename } from 'node:path';
import { apply } from '../src/index.js';
import { createTrainingHub, validate } from '../src/training-hub.js';
import { createAnimeBackend, bridgeAnime } from '../src/anime-backend.js';
import { loadTrainingConfig } from '../src/training-config.js';
import { createHash } from 'node:crypto';

const root = mkdtempSync(join(tmpdir(), 'dsh-training-integration-'));
const logger = { info() {}, warn() {}, error() {} };
const ctx = { logger, get() { return { register() {} }; } };
try {
  const manifest = JSON.parse(readFileSync(new URL('../vendor/manifest.json', import.meta.url), 'utf8'));
  for (const source of manifest.sources) for (const f of source.files) {
    const data = readFileSync(new URL(`../vendor/${source.id}/${f.path}`, import.meta.url));
    assert.equal(createHash('sha256').update(data).digest('hex'), f.patchedSha256 || f.sha256, `vendor hash: ${source.id}/${f.path}`);
  }
  const localConfig = join(root, 'training.local.json');
  writeFileSync(localConfig, JSON.stringify({ workRoot: 'local-work', pipelines: { gsv: { version: 'v2' } } }));
  const loaded = loadTrainingConfig({ trainingConfig: localConfig, workRoot: '', pipelines: { gsv: { expName: 'kept', version: 'v1' } } });
  assert.equal(loaded.workRoot, 'local-work');
  assert.equal(loaded.pipelines.gsv.version, 'v2');
  assert.equal(loaded.pipelines.gsv.expName, 'kept');
  writeFileSync(localConfig, 'not json');
  assert.throws(() => loadTrainingConfig({ trainingConfig: localConfig }), /读取失败/);
  const registered = new Map();
  await apply({ logger, get() { return { register(d) { assert(!registered.has(d.name)); registered.set(d.name, d); } }; } }, { workRoot: root });
  assert.equal(registered.size, 7);
  assert(!registered.has('train_infer') && !registered.has('tts_infer'));
  const hub = await createTrainingHub(ctx, { workRoot: root, songRoot: root, pipelines: { gsv: { version: 'v2' } } });
  const cat = hub.catalog();
  assert(cat.ok);
  assert.deepEqual(cat.routes.map(r => r.route), ['ddsp', 'gsv', 'anime']);
  for (const route of cat.routes) assert(!route.actions.some(a => /infer|cover|publish/.test(a.action)));
  assert(hub.catalog({ route: 'gsv', action: 'build' }).routes[0].actions[0].parameters.properties.lines);
  assert(!hub.catalog({ route: 'gsv', action: 'infer' }).ok);
  const before = existsSync(join(root, 'advanced-training'));
  const dry = await hub.execute({ route: 'gsv', action: 'build', params: { voiceDir: root, speaker: '角色', voiceOnly: true }, dryRun: true });
  assert(dry.ok && dry.executed === false);
  assert.equal(existsSync(join(root, 'advanced-training')), before, 'dry-run must not create receipts');
  for (const request of [
    { route: 'rvc', action: 'run' },
    { route: 'gsv', action: 'infer' },
    { route: 'ddsp', action: 'build', params: { input: root } },
    { route: 'gsv', action: 'build', params: { input: root } },
    { route: 'ddsp', action: 'run', params: { confirmOverwrite: 'true' } },
    { route: 'gsv', action: 'run', params: { s2Epochs: NaN } },
  ]) {
    const out = await hub.execute(request);
    assert.equal(out.ok, false, JSON.stringify(request));
    assert.equal(out.executed, false);
  }
  assert(validate({ type: 'object', additionalProperties: true }, JSON.parse('{"__proto__":{}}')).length);
  const plan = await hub.execute({ route: 'gsv', action: 'plan' });
  assert(plan.ok && plan.executed && plan.backendTool === 'tts_plan');
  const saved = JSON.parse(readFileSync(plan.receipt, 'utf8'));
  assert.equal(saved.result.ok, true);
  assert.equal(saved.backendTool, 'tts_plan');
  const failed = await hub.execute({ route: 'ddsp', action: 'scan', params: { input: join(root, 'missing-source') } });
  assert.equal(failed.ok, false);
  assert(failed.receipt && JSON.parse(readFileSync(failed.receipt, 'utf8')).result.ok === false);
  const wrongRoot = await createTrainingHub(ctx, { workRoot: root, pipelines: { gsv: { gsvDir: join(root, 'missing-engine') } } });
  const guarded = await wrongRoot.execute({ route: 'gsv', action: 'patch_text', params: { apply: true } });
  assert(!guarded.ok && !guarded.executed && guarded.stage === 'configuration');
  for (const sourceType of ['audio', 'game', 'anime']) {
    for (const route of ['ddsp', 'gsv']) {
      const result = await registered.get('adv_train_plan').execute({ route, sourceType, source: '/fixture', speaker: '角色' });
      assert(result.ok && !result.executed);
      assert(result.steps.some(s => s.route === route && s.action === 'run'));
      assert(!result.steps.some(s => /infer|publish|cover/.test(s.action || '')));
      if (sourceType === 'audio' && route === 'gsv') assert(result.steps.some(s => s.action === 'asr'));
      if (sourceType === 'anime' && route === 'gsv') assert(result.steps.some(s => s.action === 'bridge'));
    }
  }
  const assets = join(root, 'anime');
  mkdirSync(join(assets, 'roles', '角色'), { recursive: true });
  const wav = join(assets, 'roles', '角色', '001.wav');
  writeFileSync(wav, Buffer.from('fixture audio'));
  const source = join(assets, 'lines.jsonl');
  writeFileSync(source, JSON.stringify({ speaker: '角色', text: '原文保留。', file: 'roles/角色/001.wav', start: 1.25 }) + '\n');
  const bridged = bridgeAnime({ input: source, out: join(root, 'bridge.jsonl'), speaker: '角色' });
  assert(bridged.ok && bridged.count === 1);
  const row = JSON.parse(readFileSync(bridged.lines, 'utf8'));
  assert.equal(row.voice_path, wav.replace(/\\/g, '/'));
  assert.equal(row.text, '原文保留。');
  assert.equal(row.start, 1.25);
  assert.throws(() => bridgeAnime({ input: source, out: bridged.lines }), /不覆盖/);
  assert.throws(() => bridgeAnime({ input: source, out: join(root, 'none.jsonl'), speaker: '其他人' }), /没有匹配/);
  const nativeList = join(assets, 'train.list');
  writeFileSync(nativeList, `${wav}|角色|zh|已有台词。\n`);
  assert(bridgeAnime({ input: nativeList, out: join(root, 'list-bridge.jsonl') }).ok);
  writeFileSync(source, JSON.stringify({ speaker: '角色', text: '缺失', file: 'missing.wav' }) + '\n');
  assert.throws(() => bridgeAnime({ input: source, out: join(root, 'missing.jsonl') }), /不存在/);
  assert(!existsSync(join(root, 'missing.jsonl')));

  const whisper = join(root, 'whisper');
  const camp = join(root, 'camp');
  mkdirSync(whisper); mkdirSync(camp);
  writeFileSync(join(whisper, 'model.bin'), 'mock');
  writeFileSync(join(camp, 'campplus_cn_common.bin'), 'mock');
  const video = join(root, 'source video.mp4'); writeFileSync(video, 'fixture');
  let invoked = 0;
  const backend = createAnimeBackend({ python: 'configured-python', whisperModel: whisper, camppModel: camp }, async request => {
    invoked += 1;
    assert.equal(request.cmd, 'configured-python');
    assert.equal(request.env.HF_HUB_OFFLINE, '1');
    assert.equal(request.args[1], 'run');
    assert.equal(request.args[2], video);
    const output = request.args[request.args.indexOf('--out-dir') + 1];
    mkdirSync(output);
    writeFileSync(join(output, 'lines.jsonl'), '{}\n');
    writeFileSync(join(output, 'gpt_sovits.list'), 'fixture\n');
    return { ok: true, exitCode: 0, stdout: 'mock adapter run', stderr: '' };
  });
  const extract = await backend.get('extract').execute({ input: video, outDir: join(root, 'new-extract'), device: 'cpu' });
  assert(extract.ok && extract.needsRoleReview && invoked === 1);
  const refused = await backend.get('extract').execute({ input: video, outDir: extract.outDir });
  assert(!refused.ok && invoked === 1, 'do not overwrite extraction outputs');
  const noArtifacts = createAnimeBackend({ whisperModel: whisper, camppModel: camp }, async () => ({ ok: true, exitCode: 0, stdout: '', stderr: '' }));
  const empty = await noArtifacts.get('extract').execute({ input: video, outDir: join(root, 'empty') });
  assert(!empty.ok, 'exit code zero without output files must fail');
  assert.equal(readdirSync(join(root, 'advanced-training', 'receipts')).length, 2);
  console.log('PASS integration: real backend registration/dispatch, schemas, receipts, scope, configuration guard, anime adapter arguments/artifacts, and text/audio bridging. Anime model execution was mocked; no GPU training ran.');
} finally {
  assert.equal(dirname(resolve(root)), resolve(tmpdir()));
  assert(basename(root).startsWith('dsh-training-integration-'));
  rmSync(root, { recursive: true, force: true });
}

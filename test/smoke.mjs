/**
 * smoke.mjs —— 契约自检：不依赖三线工程，只验证插件注册形状与 py 脚本静态健康。
 */
import { strict as assert } from 'node:assert';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const EXPECTED = ['adv_doctor', 'adv_tail_audit', 'adv_unk_patch', 'adv_ab_eval'];

const mod = await import('../src/index.js');
assert.equal(mod.name, 'advanced-training', 'name 必须是 advanced-training');
assert.deepEqual(mod.inject, ['tools'], 'inject 必须是 ["tools"]');
assert.equal(typeof mod.apply, 'function', 'apply 必须是函数');

const registered = new Map();
const ctx = {
  logger: { info() {}, warn() {}, error() {} },
  get(service) {
    if (service !== 'tools') return undefined;
    return {
      register(def) {
        assert.ok(def?.name, '工具必须有 name');
        assert.ok(def?.description, `${def?.name} 必须有 description`);
        assert.ok(def?.parameters, `${def?.name} 必须有 parameters`);
        assert.ok(def?.output?.schema, `${def?.name} 必须有 output.schema`);
        assert.equal(typeof def?.output?.render, 'function', `${def?.name} 必须有 output.render`);
        assert.equal(typeof def?.execute, 'function', `${def?.name} 必须有 execute`);
        assert.ok(!registered.has(def.name), `工具重名: ${def.name}`);
        registered.set(def.name, def);
      },
    };
  },
};

await mod.apply(ctx, {});
assert.equal(registered.size, EXPECTED.length,
  `工具数应为 ${EXPECTED.length}，实际 ${registered.size}: ${[...registered.keys()]}`);
for (const n of EXPECTED) assert.ok(registered.has(n), `缺少工具: ${n}`);

// 参数 schema：JSON Schema 壳 + 显式 additionalProperties
for (const [n, def] of registered) {
  assert.equal(def.parameters.type, 'object', `${n}.parameters.type`);
  assert.ok(def.parameters.properties, `${n}.parameters.properties 缺失`);
  assert.equal(def.parameters.additionalProperties, false, `${n} 应显式 additionalProperties:false`);
}

// 无参可调用的工具不该有必填参数
const doctorReq = registered.get('adv_doctor').parameters.required || [];
assert.equal(doctorReq.length, 0, 'adv_doctor 必须无参可调');

// render 必须能吃空值而不炸
for (const [n, def] of registered) {
  const out = def.output.render({}, {});
  assert.ok(Array.isArray(out) && out[0]?.type === 'text', `${n}.render 应返回 [{type:'text'}]`);
}

// py 脚本静态健康：语法可编译（不依赖 numpy 在场）
const here = fileURLToPath(new URL('.', import.meta.url));
const root = join(here, '..');
for (const script of ['tail_audit.py', 'asr_cer.py', 'patch_unk.py']) {
  const out = execFileSync('python', ['-c', `import ast,sys;ast.parse(open(sys.argv[1],encoding="utf-8").read());print("ok")`, join(root, 'py', script)],
    { encoding: 'utf8' });
  assert.ok(out.includes('ok'), `${script} 语法不过`);
}

console.log(`✅ smoke: ${EXPECTED.length} 个工具契约 + 3 个 py 脚本语法通过`);
console.log('   ' + [...registered.keys()].join(', '));

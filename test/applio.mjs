import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve, join, dirname, basename } from 'node:path';
import { createApplioBackend, resolveApplioConfig } from '../src/applio-backend.js';
import { createTrainingHub } from '../src/training-hub.js';
const root=mkdtempSync(join(tmpdir(),'dsh-applio-'));
try {
  const c=resolveApplioConfig({repo:root,python:process.execPath,workRoot:root,logsDir:join(root,'moved-logs')});
  assert.equal(c.workRoot,join(root,'applio'));
  assert.equal(c.logsDir,join(root,'moved-logs'));
  let launched;
  const backend=createApplioBackend({repo:root,python:process.execPath,workRoot:root,logsDir:join(root,'moved-logs')},{async spawnJob(r){
    launched=JSON.parse(readFileSync(r.args.at(-1),'utf8'));
    writeFileSync(launched.statePath,JSON.stringify({ok:true,state:'running',pid:process.pid,heartbeatAt:Date.now()/1000,jobId:launched.jobId}));
    return{pidValid:true,pid:process.pid};
  }});
  assert(backend.has('adopt')&&backend.has('resume'));
  const result=await backend.get('resume').execute({expName:'fixture',epochs:30});
  assert(result.ok && result.rendered.includes('Applio'));
  assert(launched.params.resume===true && launched.action==='run');
  assert.equal(launched.config.logsDir,join(root,'moved-logs'));
  assert.equal(launched.config.workRoot,join(root,'applio'));
  const hub=await createTrainingHub({logger:{info(){}},get(){}},{workRoot:root});
  for(const request of [
    {action:'run',params:{cleanup:true}}, {action:'resume',params:{resume:false}},
    {action:'run',params:{expName:'../escape'}}, {action:'adopt',params:{logsDir:'arbitrary'}},
    {action:'infer'}, {action:'convert'}, {action:'publish'},
  ]) {
    const r=await hub.execute({route:'applio',...request,dryRun:true});
    assert(!r.ok&&!r.executed);
  }
  const { trainingPlan }=await import('../src/training-hub.js');
  const plan=trainingPlan({route:'applio',sourceType:'experiment',source:'fixture'});
  assert(plan.ok && !plan.executed && plan.steps.some(x=>x.action==='adopt') && plan.steps.some(x=>x.action==='resume'));
  assert(!trainingPlan({route:'applio',sourceType:'experiment',source:'../other'}).ok);
  execFileSync(process.env.TEST_PYTHON||'python',[fileURLToPath(new URL('./applio.py',import.meta.url))],{stdio:'inherit',windowsHide:true});
  console.log('PASS Applio: independent route/config, direct resume forces backup path, no cleanup or arbitrary process/source commands.');
} finally {
  assert.equal(dirname(resolve(root)),resolve(tmpdir()));assert(basename(root).startsWith('dsh-applio-'));
  rmSync(root,{recursive:true,force:true});
}

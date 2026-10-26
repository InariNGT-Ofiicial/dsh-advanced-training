/** Fault injection against our own short-lived Python child, no RVC/GPU needed. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
if (process.platform !== 'win32') {
  console.log('SKIP Windows job-object crash cleanup on this platform.');
} else {
  const python = process.env.TEST_PYTHON || 'python';
  const module = fileURLToPath(new URL('../py/rvc_route.py', import.meta.url));
  const load = 'import importlib.util,sys;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location("rvc_route",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);';
  const crash = load + '\nimport subprocess,os\nwith m.OwnedProcessTree() as tree:\n p=subprocess.Popen([sys.executable,"-u","-c","import os,time;print(os.getpid(),flush=True);time.sleep(30)"],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)\n tree.attach(p)\n print(p.stdout.readline().strip(),flush=True)\n os._exit(0)\n';
  const pid = Number(execFileSync(python, ['-u', '-c', crash, module], { encoding: 'utf8', timeout: 10000, windowsHide: true }).trim());
  assert(Number.isSafeInteger(pid) && pid > 0);
  const alive = execFileSync(python, ['-c', load+'print(m.is_alive(int(sys.argv[2])))', module, String(pid)], { encoding: 'utf8', timeout: 10000, windowsHide: true }).trim();
  assert.equal(alive, 'False', 'coordinator exit must also end its own child tree');
  console.log('PASS RVC Windows process tree: forced coordinator exit ends its own child; no orphan worker, no GPU training.');
}

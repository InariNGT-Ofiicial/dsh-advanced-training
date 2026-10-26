/**
 * exec.js —— 跑 python 子进程的精简封装。
 *
 * 只做一件事：spawn、收集 stdout/stderr、从混合输出里取最后一段合法 JSON
 * （脚本常先打进度日志，最后一行才是结果）。
 */
import { spawn } from 'node:child_process';

export function lastJson(text) {
  if (!text) return null;
  const t = String(text).trim();
  try { return JSON.parse(t); } catch { /* 继续 */ }
  const lines = t.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (!lines[i].startsWith('{')) continue;
    try { return JSON.parse(lines[i]); } catch { /* 继续 */ }
  }
  return null;
}

export function clip(s, n) {
  s = String(s ?? '');
  return s.length <= n ? s : s.slice(0, n) + `…(截断，共 ${s.length} 字符)`;
}

/** 跑一个 python 脚本，返回 { code, stdout, stderr, json, tail } */
export function runPy(py, script, args, timeoutMs = 300_000) {
  return new Promise((resolve) => {
    const p = spawn(py, [script, ...args], { windowsHide: true });
    let out = '';
    let err = '';
    let done = false;
    const timer = setTimeout(() => {
      if (!done) { p.kill(); done = true; resolve({ code: -1, stdout: out, stderr: err + '\n(超时被杀)', json: null, tail: clip(out + err, 2000) }); }
    }, timeoutMs);
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.on('data', (d) => { err += d; });
    p.on('error', (e) => {
      if (done) return;
      done = true; clearTimeout(timer);
      resolve({ code: -1, stdout: out, stderr: err + String(e), json: null, tail: clip(out + err + String(e), 2000) });
    });
    p.on('close', (code) => {
      if (done) return;
      done = true; clearTimeout(timer);
      resolve({ code, stdout: out, stderr: err, json: lastJson(out), tail: clip((out + '\n' + err).trim(), 2000) });
    });
  });
}

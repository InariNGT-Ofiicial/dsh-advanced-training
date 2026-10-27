import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
execFileSync(process.env.TEST_PYTHON || 'python', [fileURLToPath(new URL('./rvc-resume.py', import.meta.url))], { stdio: 'inherit', windowsHide: true });

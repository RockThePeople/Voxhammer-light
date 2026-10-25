// T1 뷰어 화면 캡처(headless Chrome, 의존성 없음: Node 22+ 내장 WebSocket·fetch + Chrome DevTools Protocol).
// 페이지가 document.body.dataset.ready = '1' 을 세울 때까지 실제 시간으로 기다렸다가 PNG 를 저장한다
// (--screenshot 의 가상 시간 아래에서는 GLB 텍스처 디코딩이 끝나지 않아 빈 화면이 찍힌다).
//   node tools/viewer/capture.mjs <out.png> <url> [폭=1024] [높이=폭] [제한초=120]
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [out, url, w = '1024', h = w, limit = '120'] = process.argv.slice(2);
if (!out || !url) { console.error('usage: capture.mjs <out.png> <url> [w] [h] [timeout_s]'); process.exit(2); }
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(process.env.CHROME || 'google-chrome', ['--headless=new', '--no-sandbox', '--enable-unsafe-swiftshader',
  '--use-angle=swiftshader', '--hide-scrollbars', `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'vhlcap-'))}`,
  `--window-size=${w},${h}`, 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const done = (code) => { try { chrome.kill('SIGKILL'); } catch (e) {} process.exit(code); };
setTimeout(() => { console.error('timeout'); done(3); }, Number(limit) * 1000);

let tgt;
for (let i = 0; i < 100 && !tgt; i++) {
  try { tgt = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page'); } catch (e) { await sleep(100); }
}
const ws = new WebSocket(tgt.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const wait = new Map(); const logs = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && wait.has(m.id)) { wait.get(m.id)(m); wait.delete(m.id); }
  if (m.method === 'Runtime.consoleAPICalled') logs.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
  if (m.method === 'Runtime.exceptionThrown') logs.push('EXC ' + JSON.stringify(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
};
const send = (method, params = {}) => new Promise((r) => { const i = ++id; wait.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send('Runtime.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: Number(w), height: Number(h), deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url });
for (;;) {
  const r = await send('Runtime.evaluate', { expression: 'document.body && document.body.dataset.ready', returnByValue: true });
  if (r.result?.result?.value === '1') break;
  await sleep(250);
}
await sleep(300);
const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
if (logs.length) console.error(logs.join('\n'));
console.log(`saved ${out}`);
done(0);

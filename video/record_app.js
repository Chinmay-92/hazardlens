// Records the scripted HazardLens walkthrough with a CDP screencast (crisp 1920x1080 JPEG frames + timestamps).
// Each app segment is held for LEAD + narration duration + TAIL seconds so footage matches the voiceover.
// Usage: node record_app.js   (needs the app on http://localhost:8080 with an empty approval queue,
//         e.g. `docker restart hl-test`, and audio/durations.json from make_audio.sh)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const APP = process.env.APP_URL || 'http://localhost:8080';
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(__dirname, 'build', 'app');
const LEAD = 0.5, TAIL = 0.8;
const dur = JSON.parse(fs.readFileSync(path.join(__dirname, 'audio', 'durations.json'), 'utf8'));
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(path.join(OUT, 'frames'), { recursive: true });

const CURSOR = `
(() => {
  const mk = () => {
    if (document.getElementById('__cur')) return;
    const c = document.createElement('div'); c.id = '__cur';
    c.innerHTML = '<svg width="28" height="28" viewBox="0 0 24 24"><path d="M3 2l7 19 2.6-7.4L20 11z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, {position:'fixed', left:'-50px', top:'-50px', zIndex: 2147483647, pointerEvents:'none', transform:'translate(-3px,-2px)'});
    document.documentElement.appendChild(c);
    addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    addEventListener('mousedown', e => {
      const r = document.createElement('div');
      Object.assign(r.style, {position:'fixed', left:(e.clientX-18)+'px', top:(e.clientY-18)+'px', width:'36px', height:'36px',
        borderRadius:'50%', border:'3px solid #ffbf47', zIndex:2147483646, pointerEvents:'none', transition:'all .45s ease-out', opacity:'1'});
      document.documentElement.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = 'scale(1.8)'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 600);
    }, true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mk); else mk();
})();`;

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 810 }, deviceScaleFactor: 4 / 3, colorScheme: 'light' });
  await ctx.addInitScript(CURSOR);
  const page = await ctx.newPage();
  const cdp = await ctx.newCDPSession(page);
  const frames = [];
  let n = 0;
  cdp.on('Page.screencastFrame', async f => {
    const file = `f${String(n++).padStart(6, '0')}.jpg`;
    fs.writeFileSync(path.join(OUT, 'frames', file), Buffer.from(f.data, 'base64'));
    frames.push({ file, t: f.metadata.timestamp });
    cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId }).catch(() => {});
  });

  await page.goto(APP, { waitUntil: 'networkidle' });
  await page.waitForTimeout(800);
  await page.mouse.move(700, 120);
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
  // Nudge a repaint so there is always an initial frame.
  await page.evaluate(() => { document.body.style.outline = '0px solid transparent'; });
  await page.waitForTimeout(600);

  const segments = [];
  let seg = null;
  const now = () => Date.now() / 1000;
  const begin = id => { seg = { id, start: now(), len: LEAD + dur[id] + TAIL }; };
  const at = async t => { const w = (seg.start + LEAD + t - now()) * 1000; if (w > 0) await page.waitForTimeout(w); };
  const end = async () => { const w = (seg.start + seg.len - now()) * 1000; if (w > 0) await page.waitForTimeout(w); seg.end = seg.start + seg.len; segments.push(seg); };

  let mx = 700, my = 120;
  const moveTo = async (x, y, steps = 28) => { await page.mouse.move(x, y, { steps }); mx = x; my = y; };
  const boxOf = async sel => { const el = page.locator(sel).first(); await el.scrollIntoViewIfNeeded().catch(() => {}); return el.boundingBox(); };
  const hover = async (sel, dx = 0.5, dy = 0.5) => { const b = await boxOf(sel); if (b) await moveTo(b.x + b.width * dx, b.y + b.height * dy); };
  const click = async sel => { await hover(sel); await page.waitForTimeout(150); await page.mouse.down(); await page.waitForTimeout(80); await page.mouse.up(); };
  const scrollTo = async y => { await page.evaluate(y => window.scrollTo({ top: y, behavior: 'smooth' }), y); await page.waitForTimeout(700); };
  const runSample = async idx => { await click(`#samples button:nth-child(${idx})`); await page.waitForFunction(() => !document.querySelector('#go').disabled && document.querySelector('#status').textContent.startsWith('Run')); };
  const traceStep = n => `#trace li:nth-child(${n}) .tool`;
  const traceN = async tool => page.evaluate(t => [...document.querySelectorAll('#trace li .tool')].findIndex(e => e.textContent === t) + 1, tool);
  const traceNth = async (tool, k = 1) => page.evaluate(([t, k]) => { let c = 0; const l = [...document.querySelectorAll('#trace li .tool')]; for (let i = 0; i < l.length; i++) if (l[i].textContent === t && ++c === k) return i + 1; return 0; }, [tool, k]);

  // 02 problem: tour the page
  begin('02_problem');
  await at(0.5); await hover('header h1');
  await at(3); await hover('header p', 0.6, 0.5);
  await at(7); await hover('#samples button:nth-child(3)');
  await at(9); await hover('#samples button:nth-child(4)');
  await at(11); await hover('#samples button:nth-child(5)');
  await at(13); await hover('#samples button:nth-child(1)');
  await at(16); await hover('#h-res');
  await at(19); await hover('#h-ap');
  await end();

  // 03 upload a spill photo
  begin('03_upload_spill');
  await at(0.3); await click('#file');
  await page.setInputFiles('#file', path.join(ROOT, 'data/samples/scene_026.jpg'));
  await at(2.2); await click('#loc'); await page.keyboard.type('Corridor B, outside room 12', { delay: 45 });
  await at(5.6); await click('#go');
  await page.waitForSelector('#result:not([hidden])');
  await at(7.5); await hover(traceStep(1), 0.5, 0.5);
  await at(11); await hover(traceStep(2));
  await at(13); await hover('#annotated', 0.27, 0.27);
  await at(15); await page.evaluate(() => window.scrollTo({ top: 140, behavior: 'smooth' })); await hover(traceStep(3));
  await at(19); await hover('#dets tr td:nth-child(3)');
  await at(23); await hover(traceStep(4));
  await at(26.5); await hover('#outcome', 0.4, 0.5);
  await end();

  // 04 shadow scene
  begin('04_shadow');
  await scrollTo(0);
  await at(0.4); await runSample(1);
  await at(3); await hover(traceStep(2));
  await at(6.5); await page.evaluate(() => window.scrollTo({ top: 120, behavior: 'smooth' })); await hover(traceStep(3));
  await at(9.5); await hover('#dets tr td:nth-child(3)');
  await at(12.5); await hover('#outcome', 0.4, 0.5);
  await end();

  // 05 dark cable scene
  begin('05_dark_cable');
  await scrollTo(0);
  await at(0.3); await runSample(2);
  await at(2); await hover(traceStep(1));
  await at(4); await hover(traceStep(2));
  await at(7); await hover('#annotated', 0.62, 0.12);
  await at(9.5); await page.evaluate(() => window.scrollTo({ top: 200, behavior: 'smooth' })); await hover(traceStep(await traceNth('detect_hazards', 2)));
  await at(18); await hover(traceStep(await traceN('inspect_region')));
  await at(22); await hover(traceStep(await traceN('request_human_approval')));
  await end();

  // 06 blurry scene
  begin('06_blurry');
  await scrollTo(0);
  await at(0.3); await runSample(4);
  await at(2.5); await hover(traceStep(1));
  await at(6.5); await hover(traceStep(2));
  await at(10); await hover('#outcome', 0.4, 0.5);
  await end();

  // 07 clutter scene
  begin('07_clutter');
  await at(0.2); await runSample(5);
  await at(2.5); await hover('#annotated', 0.3, 0.6);
  await at(5); await hover(traceStep(await traceN('create_work_order')));
  await at(7.5); await hover('#outcome', 0.4, 0.5);
  await end();

  // 08 approve the cable alert
  begin('08_approve');
  await at(0.3); await page.evaluate(() => window.scrollTo({ top: 760, behavior: 'smooth' })); await page.waitForTimeout(800);
  await hover('#h-ap');
  await at(3.2); await hover('#approvals .card:nth-child(1)', 0.5, 0.3);
  await at(5); await hover('#approvals .card:nth-child(1) button');
  await at(6.6); await click('#approvals .card:nth-child(1) button');
  await page.waitForFunction(() => document.querySelector('#approvals').textContent.includes('Last alert'));
  await at(8.5); await hover('#approvals .card:nth-child(1)', 0.5, 0.25);
  await at(10.5); await hover('#approvals p.muted', 0.3, 0.5);
  await end();

  await page.waitForTimeout(400);
  await cdp.send('Page.stopScreencast');
  fs.writeFileSync(path.join(OUT, 'timeline.json'), JSON.stringify({ lead: LEAD, tail: TAIL, segments, frames }, null, 1));
  console.log('frames', frames.length, 'segments', segments.map(s => `${s.id}:${s.len.toFixed(1)}`).join(' '));
  await browser.close();
})();

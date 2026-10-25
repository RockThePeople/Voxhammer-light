// T1 시나리오 페이지: 조건 이미지, 카메라 연동 3D 비교, 복셀 겹치기, 성분 표, 수치 표.
// 좌표 변환·카메라는 common.js 의 voxelToWorld·placeNormalized·forkCamera 에서만 한다.
import { THREE, loadGLB, addLights, prepare, voxelToWorld, placeNormalized, forkCamera } from '/common.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const sid = decodeURIComponent(location.pathname.split('/').pop());
const Q = new URLSearchParams(location.search);
const CAPTURE = Q.get('capture');            // 'edit' | '34' | '0'..'4' → 1024² 한 칸만, 초록 배경(V-2 실루엣)
const sfile = (key) => `/sfile?id=${encodeURIComponent(sid)}&key=${encodeURIComponent(key)}`;
const OV = [
  ['mask', '마스크', 0xff2020], ['halo', 'halo 2 고리', 0xff9900], ['new_in', '마스크∪halo 안 신규', 0x18b418],
  ['new_out', 'R_keep 신규(9a 남음)', 0xffd400], ['new_out_9a', 'R_keep 신규(9a 제거 대상)', 0xb8860b],
  ['lost', '소실', 0x2060ff], ['sel', '선택한 성분', 0xff00ff],
];
// halo 고리는 큰 마스크에서 화면을 덮으므로 기본 꺼짐(체크박스로 켠다)
const state = { meta: null, panels: [], ov: new Set((Q.get('ov') ?? 'mask,new_in,new_out,new_out_9a,lost').split(',').filter(Boolean)),
  mode: Q.get('mode') || 'both', selComp: null, compPanel: null };

// 공유 카메라: 모든 칸이 같은 카메라 객체를 쓰고, 각 칸의 OrbitControls 가 같은 target 으로 그것을 움직인다.
const camera = new THREE.PerspectiveCamera(49.13, 1, 0.01, 100);
const target = new THREE.Vector3(0, 0, 0);
const boxGeo = new THREE.BoxGeometry(1, 1, 1);

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => (k === 'class' ? (e.className = v) : k.startsWith('on') ? (e[k] = v) : e.setAttribute(k, v)));
  kids.flat().forEach((k) => e.append(k instanceof Node ? k : document.createTextNode(String(k))));
  return e;
}
const fmt = (v, d = 4) => (v == null || Number.isNaN(v) ? '–' : typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(d)) : v);

function renderAll() { state.panels.forEach((p) => p.renderer && p.renderer.render(p.scene, camera)); }

function setView(v) {
  const m = state.meta;
  const view = v === 'edit' ? m.edit_view : v === '34' ? m.view_34 : Number(v);
  if (view == null || Number.isNaN(view)) return;
  forkCamera(camera, view, m.camera);
  target.set(0, 0, 0);
  state.panels.forEach((p) => p.controls && p.controls.update());
  document.querySelectorAll('#camBar button').forEach((b) => b.classList.toggle('on', b.dataset.v === String(v)));
  renderAll();
}

function instanced(flat, xf, color, kind) {
  const n = flat.length / 3;
  if (!n) return null;
  const solid = Q.get('solid') === '1';
  const mat = kind === 'halo' && !solid
    ? new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.45 })
    : kind === 'mask' && !solid
      ? new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18, depthWrite: false })
      : new THREE.MeshLambertMaterial({ color: solid ? 0xffffff : color });
  if (solid) mat.emissive = new THREE.Color(0xffffff);
  const mesh = new THREE.InstancedMesh(boxGeo, mat, n);
  const pos = voxelToWorld(flat, xf), m4 = new THREE.Matrix4(), s = (xf.scale / 64) * (kind === 'halo' ? 0.9 : 0.96);
  for (let i = 0; i < n; i++) { m4.makeScale(s, s, s).setPosition(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]); mesh.setMatrixAt(i, m4); }
  mesh.renderOrder = kind === 'mask' ? 2 : 1;
  mesh.userData.kind = kind;
  return mesh;
}

function applyVisibility(p) {
  if (p.mesh) p.mesh.visible = state.mode !== 'voxel';
  p.ovGroup.children.forEach((o) => {
    const k = o.userData.kind;
    o.visible = state.mode !== 'mesh' && (k === 'sel' ? state.selComp != null : state.ov.has(k));
  });
}

async function buildPanel(p) {
  const m = state.meta;
  p.ovGroup.clear();
  if (p.mesh) { p.scene.remove(p.mesh); p.mesh = null; }
  p.legend.textContent = '불러오는 중…';
  const add = (o) => o && p.ovGroup.add(o);
  add(instanced(m.mask_cells, m.xf, 0xff2020, 'mask'));
  add(instanced(m.halo_cells, m.xf, 0xff9900, 'halo'));
  const legend = [];
  legend.push([0xff2020, `마스크 ${m.mask.n_cells}`]);
  p.vox = null;
  if (p.cfg !== 'src') {
    const v = await (await fetch(`/api/scenario_vox?id=${encodeURIComponent(sid)}&cfg=${p.cfg}&run=${p.run}`)).json();
    p.vox = v;
    const out9a = [], outKeep = [];
    v.components.forEach((c) => (c.removed_by_9a ? out9a : outKeep).push(...c.cells));
    add(instanced(v.new_in, m.xf, 0x18b418, 'new_in'));
    add(instanced(outKeep, m.xf, 0xffd400, 'new_out'));
    add(instanced(out9a, m.xf, 0xb8860b, 'new_out_9a'));
    add(instanced([...v.lost_in, ...v.lost_out], m.xf, 0x2060ff, 'lost'));
    if (state.selComp != null && state.compPanel === p && v.components[state.selComp]) {
      const sel = instanced(v.components[state.selComp].cells, m.xf, 0xff00ff, 'sel');
      if (sel) { sel.scale.setScalar(1); p.ovGroup.add(sel); }
    }
    const c = v.counts;
    legend.push([0x18b418, `안 신규 ${c.new_in}`], [0xffd400, `R_keep 신규 ${c.new_out} (9a 제거 ${v.removed_by_9a_cells})`],
                [0x2060ff, `소실 ${c.lost_in + c.lost_out} (R_keep ${c.lost_out})`]);
  }
  try {
    const url = p.cfg === 'src' ? sfile('src_glb') : sfile(`glb:${p.cfg}:${p.run}`);
    const obj = await loadGLB(url);
    prepare(obj);
    // TRELLIS to_glb 재질은 금속성이라 환경맵 없이 검게 보인다 → 이 페이지에서만 비금속으로(형태 비교용)
    obj.traverse((o) => o.isMesh && [o.material].flat().forEach((mt) => { if (mt && 'metalness' in mt) { mt.metalness = 0; mt.roughness = Math.max(mt.roughness ?? 1, 0.7); } }));
    if (p.cfg !== 'src' && m.runs[p.cfg][p.run].normalize) placeNormalized(obj, m.xf);
    p.mesh = obj;
    p.scene.add(obj);
  } catch (e) { legend.push([0x000000, `GLB 실패: ${e}`]); }
  applyVisibility(p);
  p.legend.replaceChildren(...legend.map(([c, t]) => el('span', {}, el('span', { class: 'sw', style: `background:#${c.toString(16).padStart(6, '0')}` }), t, ' ')));
  renderAll();
  if (state.compPanel === p || (!state.compPanel && p.cfg !== 'src')) showComps(p);
}

function panelChoices() {
  const m = state.meta, out = [['src', '원본']];
  Object.entries(m.configs).forEach(([c, d]) => out.push([c, `${c} — ${d}`]));
  return out;
}

function addPanel(cfg = 'src', run = 'r1', host = document.getElementById('panels'), size = null) {
  const m = state.meta;
  const p = { cfg, run, scene: new THREE.Scene(), ovGroup: new THREE.Group(), mesh: null };
  p.scene.background = new THREE.Color(CAPTURE ? 0x00ff00 : 0xf7f7f9);
  if (!(CAPTURE && Q.get('lights') === '0')) addLights(p.scene);
  p.scene.add(p.ovGroup);
  const canvas = el('canvas');
  p.renderer = new THREE.WebGLRenderer({ canvas, antialias: !CAPTURE, preserveDrawingBuffer: true });
  p.renderer.setPixelRatio(CAPTURE ? 1 : window.devicePixelRatio);
  p.legend = el('div', { class: 'legend' });
  if (CAPTURE) {
    host.append(canvas);
    p.renderer.setSize(size, size, false);
  } else {
    const cfgSel = el('select', {}, ...panelChoices().map(([v, t]) => el('option', { value: v, ...(v === cfg ? { selected: '' } : {}) }, t)));
    const runSel = el('select', {}, ...['r1', 'r2', 'r3'].map((r) => el('option', { value: r, ...(r === run ? { selected: '' } : {}) }, r)));
    const sync = () => {
      const runs = cfgSel.value === 'src' ? [] : Object.keys(m.runs[cfgSel.value] || {});
      [...runSel.options].forEach((o) => (o.disabled = cfgSel.value !== 'src' && !runs.includes(o.value)));
      runSel.disabled = cfgSel.value === 'src';
      if (cfgSel.value !== 'src' && !runs.includes(runSel.value)) runSel.value = runs[0];
    };
    sync();
    cfgSel.onchange = runSel.onchange = () => { sync(); p.cfg = cfgSel.value; p.run = runSel.value; buildPanel(p); };
    const close = el('span', { class: 'x', title: '칸 닫기', onclick: () => removePanel(p) }, '✕');
    const compBtn = el('button', { title: '이 칸의 성분 표 보기', onclick: () => showComps(p) }, '성분');
    // V-5: R4 결과 칸은 "청크" 보기(부모 청크 위에 변경 청크만 교체)와 "교체 전/후" 토글
    p.chunkBtn = el('button', { title: '부모 청크 위에 변경 청크만 교체해 보기', onclick: () => toggleChunks(p) }, '청크');
    p.swapBtn = el('button', { title: '교체 전 / 교체 후', onclick: () => swapChunks(p) }, '교체 후');
    p.swapBtn.hidden = true;
    const syncChunkBtn = () => { p.chunkBtn.hidden = !(m.runs[p.cfg]?.[p.run]?.chunk_view); };
    cfgSel.addEventListener('change', syncChunkBtn); runSel.addEventListener('change', syncChunkBtn); syncChunkBtn();
    p.root = el('div', { class: 'panel' }, el('div', { class: 'ph' }, cfgSel, runSel, compBtn, p.chunkBtn, p.swapBtn, close), canvas, p.legend);
    host.append(p.root);
    p.controls = new OrbitControls(camera, canvas);
    p.controls.target = target;   // 모든 칸이 같은 target 객체를 공유
    p.controls.enableDamping = false;
    p.controls.addEventListener('change', renderAll);
    new ResizeObserver(() => { const w = canvas.clientWidth; if (w) { p.renderer.setSize(w, w, false); renderAll(); } }).observe(canvas);
  }
  state.panels.push(p);
  buildPanel(p);
  return p;
}

// ── V-5 청크 보기 ─────────────────────────────────────────────────────────────
const CHUNK_COLORS = { inherited: 0x9a9aa6, book: 0xff2020, occupancy: 0xffc400, removed: 0x555555 };
const CHUNK_LABEL = { inherited: '승계(부모 바이트)', book: '변경 — 부기(마스크 ∪ halo)', occupancy: '변경 — 점유 변경으로만', removed: '제거' };

function chunkBox(key, xf, color, dashed) {
  const [x, y, z] = key.split('_').map(Number);
  const c = voxelToWorld([4 * x + 1.5, 4 * y + 1.5, 4 * z + 1.5], xf), sz = 4 * xf.scale / 64;
  const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(sz, sz, sz));
  const mat = dashed ? new THREE.LineDashedMaterial({ color, dashSize: sz / 6, gapSize: sz / 8 })
    : new THREE.LineBasicMaterial({ color, transparent: color === CHUNK_COLORS.inherited, opacity: 0.35 });
  const l = new THREE.LineSegments(g, mat);
  l.position.set(c[0], c[1], c[2]);
  if (dashed) l.computeLineDistances();
  return l;
}

async function toggleChunks(p) {
  const m = state.meta;
  if (p.chunk) {   // 끄기 → 원래 칸으로
    p.scene.remove(p.chunk.group); p.chunk = null; p.swapBtn.hidden = true; p.chunkBtn.classList.remove('on');
    if (p.mesh) p.mesh.visible = state.mode !== 'voxel';
    p.ovGroup.visible = true; document.getElementById('sec-chunks').hidden = true; renderAll(); return;
  }
  p.chunkBtn.classList.add('on'); p.legend.textContent = '청크 불러오는 중…';
  const info = await (await fetch(`/api/scenario_chunks?id=${encodeURIComponent(sid)}&cfg=${p.cfg}&run=${p.run}`)).json();
  const [par, chg] = await Promise.all([loadGLB(sfile(`chunks_parent:${p.cfg}:${p.run}`)), loadGLB(sfile(`chunks_changed:${p.cfg}:${p.run}`))]);
  [par, chg].forEach((o) => { prepare(o); o.traverse((x) => x.isMesh && [x.material].flat().forEach((mt) => { if (mt && 'metalness' in mt) { mt.metalness = 0; mt.roughness = 0.8; } }));
    if (m.runs[p.cfg][p.run].normalize) placeNormalized(o, m.xf); });
  const boxes = new THREE.Group();
  const state_ = Object.fromEntries(info.rows.map((r) => [r.key, r.state]));
  info.rows.forEach((r) => boxes.add(chunkBox(r.key, m.xf, CHUNK_COLORS[r.state], r.state === 'removed')));
  const group = new THREE.Group(); group.add(par, chg, boxes);
  p.chunk = { group, par, chg, state: state_, after: true, info };
  p.scene.add(group);
  if (p.mesh) p.mesh.visible = false;
  p.ovGroup.visible = false;
  p.swapBtn.hidden = false;
  applySwap(p);
  showChunkTable(p);
}

function nodeKey(o) { let x = o; while (x && !/^\d+_\d+_\d+$/.test(x.name)) x = x.parent; return x ? x.name : null; }

function applySwap(p) {
  const c = p.chunk;
  // 교체 전 = 부모 청크 전부. 교체 후 = 부모 중 승계 청크만 남기고, 변경 청크 자리에 자식 청크를 넣는다(제거 청크는 비운다).
  c.par.traverse((o) => { if (o.isMesh) { const k = nodeKey(o); o.visible = !c.after || c.state[k] === 'inherited'; } });
  c.chg.visible = c.after;
  p.swapBtn.textContent = c.after ? '교체 후 (→ 교체 전)' : '교체 전 (→ 교체 후)';
  const n = c.info.counts;
  p.legend.replaceChildren(...Object.entries(CHUNK_LABEL).map(([k, t]) => el('span', {},
    el('span', { class: 'sw', style: `background:#${CHUNK_COLORS[k].toString(16).padStart(6, '0')}` }), `${t} ${n[k]} `)));
  renderAll();
}

function swapChunks(p) { if (p.chunk) { p.chunk.after = !p.chunk.after; applySwap(p); } }

function showChunkTable(p) {
  const info = p.chunk.info, sec = document.getElementById('sec-chunks');
  sec.hidden = false;
  document.getElementById('chunksFor').textContent = `${p.cfg} ${p.run}`;
  document.getElementById('chunksNote').textContent = `${info.rule}. 출처: ${info.source}. `
    + `승계 청크 전부 부모와 바이트 동일: ${info.inherited_all_equal_parent ? '예' : '아니오'}. `
    + `부모 ${info.parent_dir} · 자식 ${info.child_dir}. 변경·제거 청크를 위로 정렬.`;
  const order = { book: 0, occupancy: 1, removed: 2, inherited: 3 };
  const rows = [...info.rows].sort((a, b) => order[a.state] - order[b.state] || a.key.localeCompare(b.key));
  const head = el('tr', {}, ...['청크 키', '상태', '면 수', '부모 면 수', '바이트', 'sha256', '부모와 바이트 동일'].map((h) => el('th', {}, h)));
  document.getElementById('chunkTable').replaceChildren(head, ...rows.map((r) => el('tr', {},
    el('td', { class: 'l' }, r.key), el('td', { class: 'l', style: `color:#${CHUNK_COLORS[r.state].toString(16).padStart(6, '0')}` }, CHUNK_LABEL[r.state]),
    el('td', {}, r.faces), el('td', {}, r.parent_faces ?? '–'), el('td', {}, r.bytes), el('td', { class: 'l' }, r.sha256),
    el('td', { class: r.state === 'inherited' ? (r.equal_parent ? 'l good' : 'l bad') : 'l' }, r.state === 'inherited' ? (r.equal_parent ? '예' : '아니오') : '–'))));
}

function removePanel(p) {
  state.panels = state.panels.filter((x) => x !== p);
  p.controls && p.controls.dispose();
  p.renderer.dispose();
  p.root && p.root.remove();
  if (state.compPanel === p) { state.compPanel = null; state.selComp = null; }
  layout();
}

function layout() { document.getElementById('panels').style.setProperty('--cols', Math.min(Math.max(state.panels.length, 1), 3)); }

function showComps(p) {
  state.compPanel = p;
  const t = document.getElementById('comps');
  document.getElementById('compsFor').textContent = p.cfg === 'src' ? '원본 칸에는 없음' : `${p.cfg} ${p.run}`;
  if (!p.vox) { t.replaceChildren(); return; }
  const head = el('tr', {}, ...['#', '셀 수', '거리(체비셰프)', '거리(유클리드)', '맞닿음', '9a 대상'].map((h) => el('th', {}, h)));
  const rows = p.vox.components.map((c, i) => el('tr', { class: state.selComp === i ? 'sel' : '', onclick: () => {
    state.selComp = state.selComp === i ? null : i; buildPanel(p); } },
  el('td', {}, i), el('td', {}, c.n), el('td', {}, c.dist_cheb), el('td', {}, c.dist_euclid), el('td', { class: 'l' }, c.touch ? '예' : '아니오'),
  el('td', { class: c.removed_by_9a ? 'l good' : 'l' }, c.removed_by_9a ? '예(제거)' : '아니오')));
  t.replaceChildren(head, ...rows);
}

function header(m) {
  document.title = `${m.id} — 시나리오 비교`;
  document.getElementById('title').textContent = `${m.id}  ${m.prompt}`;
  if (m.e12) document.getElementById('tags').append(el('span', { class: 'tag' }, m.e12));
  const kv = [
    ['트랙', m.track], ['데이터셋·객체', `${m.dataset} / ${m.object}`], [m.prompt_kind, m.prompt],
    ...(m.source_prompt ? [['원본 프롬프트', m.source_prompt]] : []),
    ['마스크', `${m.mask.n_cells} 셀 · 지문 ${String(m.mask.fingerprint).slice(0, 12)} · 파일 sha ${m.mask.file_sha16}`],
    ['편집 시점 / 3/4 시점', `${m.edit_view ?? '–'} / ${m.view_34 ?? '–'}`],
    ['원본 SLat 셀', m.source_cells_n], ['좌표 정규화', `c = [${m.xf.center.map((x) => x.toFixed(5)).join(', ')}], s = ${m.xf.scale.toFixed(5)} (${m.xf.src})`],
    ...(m.alignment ? [['정렬(V-2)', `원본 셀 중심 → 원본 메시 1셀 이내 ${(100 * m.alignment.within_1cell).toFixed(1)}%`
      + (m.alignment.decoder_mesh_within_1cell != null ? ` · 디코더 메시(to_glb 전) ${(100 * m.alignment.decoder_mesh_within_1cell).toFixed(1)}%` : '')]] : []),
  ];
  document.getElementById('head').replaceChildren(...kv.flatMap(([k, v]) => [el('b', {}, k), el('span', {}, v)]));
}

async function conds(m) {
  const box = document.getElementById('conds');
  const figs = [];
  for (const n of ['2d_render', '2d_mask', '2d_edit']) {
    if (!m.cond[n]) continue;
    const img = el('img', { src: sfile(`cond:${n}`), alt: n, loading: 'lazy' });
    img.onclick = () => window.open(img.src, '_blank');
    figs.push(el('figure', {}, img, el('figcaption', {}, n)));
  }
  if (m.cond['2d_render'] && m.cond['2d_mask']) {
    const cv = el('canvas', { width: 1024, height: 1024 });
    const [a, b] = await Promise.all(['2d_render', '2d_mask'].map((n) => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = sfile(`cond:${n}`); })));
    const g = cv.getContext('2d');
    g.fillStyle = '#fff'; g.fillRect(0, 0, 1024, 1024); g.drawImage(a, 0, 0, 1024, 1024);
    const t = document.createElement('canvas'); t.width = t.height = 1024; const tg = t.getContext('2d');
    tg.drawImage(b, 0, 0, 1024, 1024);
    const d = tg.getImageData(0, 0, 1024, 1024);
    for (let i = 0; i < d.data.length; i += 4) {
      const on = d.data[i + 3] > 0 && (d.data[i] + d.data[i + 1] + d.data[i + 2]) > 128 * 3;
      d.data[i] = 255; d.data[i + 1] = 0; d.data[i + 2] = 0; d.data[i + 3] = on ? 110 : 0;
    }
    tg.putImageData(d, 0, 0); g.drawImage(t, 0, 0);
    cv.onclick = () => window.open(cv.toDataURL('image/png'), '_blank');
    figs.push(el('figure', {}, cv, el('figcaption', {}, '2d_render + 2d_mask(반투명 빨강)')));
  }
  box.replaceChildren(...figs);
}

function metrics(m) {
  const T = m.metrics, rows = T.rows, cfgs = Object.keys(rows);
  const mk = ['psnr', 'ssim', 'lpips', 'chamfer', 'dino_i', 'clip_t'].filter((k) => cfgs.some((c) => rows[c].metrics && k in rows[c].metrics));
  const head = el('tr', {}, el('th', { class: 'l' }, '설정'), el('th', {}, 'R_keep 소실'), el('th', {}, 'R_keep 신규'), el('th', {}, '편집량'),
    ...mk.map((k) => el('th', {}, k)), el('th', {}, '구간 4–7 (s)'), el('th', { class: 'l' }, 'E19'));
  const body = cfgs.map((c) => {
    const r = rows[c], e = r.e19 ? r.e19.cells : {};
    const ex = Object.values(e).filter((x) => x.exceed).length, bt = Object.values(e).filter((x) => x.better).length;
    return el('tr', {}, el('td', { class: 'l' }, `${c} — ${m.configs[c] || ''}`), el('td', {}, fmt(r.rkeep_lost, 0)), el('td', {}, fmt(r.rkeep_new, 0)),
      el('td', {}, fmt(r.edit_amount, 0)),
      ...mk.map((k) => el('td', { class: e[k] ? (e[k].exceed ? 'bad' : '') : '', title: e[k] ? `Δ vs ${r.e19.ref} = ${e[k].delta.toFixed(4)}` : '' },
        r.metrics && k in r.metrics ? fmt(r.metrics[k]) : '–')),
      el('td', {}, r.seg4_7 ? `${r.seg4_7[0].toFixed(1)} ± ${r.seg4_7[1].toFixed(1)}` : '–'),
      el('td', { class: 'l' }, r.e19 ? `기준 ${r.e19.ref}: 초과 ${ex} · 개선 ${bt}` : '–'));
  });
  const src = Object.entries(T.sources).map(([k, v]) => el('div', { class: 'note' }, `${k}: ${v.path} (sha256 ${v.sha256_16}…)`));
  document.getElementById('metrics').replaceChildren(el('table', {}, head, ...body),
    el('div', { class: 'note' }, `${T.e19_note} 허용치: ${Object.entries(T.e19_tolerance).map(([k, v]) => `${k} ${v > 0 ? '≤ +' : '≥ '}${v}`).join(', ')}. 칸에 마우스를 올리면 Δ.`),
    ...src);
}

function bars(m) {
  const cb = document.getElementById('camBar');
  const views = [['edit', `편집 시점 (${m.edit_view ?? '–'})`], ['34', `3/4 시점 (${m.view_34 ?? '–'})`], ...[0, 1, 2, 3, 4].map((v) => [String(v), `포크 ${v}`])];
  cb.replaceChildren(el('b', {}, '카메라'), ...views.map(([v, t]) => el('button', { 'data-v': v, onclick: () => setView(v) }, t)),
    el('button', { onclick: () => addPanel('c15', 'r1') && layout() }, '+ 칸 추가'));
  const ob = document.getElementById('ovBar');
  const mode = el('select', { onchange: (e) => { state.mode = e.target.value; state.panels.forEach(applyVisibility); renderAll(); } },
    ...[['both', '메시 + 복셀'], ['mesh', '메시만'], ['voxel', '복셀만']].map(([v, t]) => el('option', { value: v, ...(v === state.mode ? { selected: '' } : {}) }, t)));
  ob.replaceChildren(el('b', {}, '겹치기'), ...OV.filter(([k]) => k !== 'sel').map(([k, t, c]) => {
    const box = el('input', { type: 'checkbox', ...(state.ov.has(k) ? { checked: '' } : {}) });
    box.onchange = () => { box.checked ? state.ov.add(k) : state.ov.delete(k); state.panels.forEach(applyVisibility); renderAll(); };
    return el('label', {}, box, el('span', { class: 'sw', style: `background:#${c.toString(16).padStart(6, '0')}` }), t);
  }), mode);
  document.getElementById('note3d').textContent =
    `좌표: 셀 → ((i,j,k)+0.5)/64 − 0.5 → glTF (x, z, −y) → (p − c) × s. 카메라 버튼 = 포크 조건 렌더와 같은 정의(반지름 1.8, 고도 15°, 화각 49.13°). `
    + `결과 GLB 는 원본과 같은 정규화로 놓는다. 큰 GLB 는 칸을 줄여 브라우저 메모리를 아낀다(최대 6칸 권장).`;
}

async function main() {
  const m = await (await fetch(`/api/scenario?id=${encodeURIComponent(sid)}`)).json();
  if (m.error) { document.body.textContent = `시나리오 ${sid}: ${m.error}`; return; }
  state.meta = m;
  if (CAPTURE) {
    document.body.classList.add('capture');
    const [cfg, run] = (Q.get('panel') || 'src').split(':');
    const p = addPanel(cfg, run || 'r1', document.getElementById('capture'), Number(Q.get('size') || 1024));
    await waitBuilt(p);
    setView(CAPTURE);
    document.body.dataset.ready = '1';
    return;
  }
  if (Q.get('only3d') === '1') document.body.classList.add('only3d');
  header(m); conds(m); bars(m); metrics(m);
  const want = (Q.get('panels') || `src,base:r1,c15:r1`).split(',').filter(Boolean);
  const ps = want.map((w) => { const [c, r] = w.split(':'); return (c === 'src' || m.runs[c]) ? addPanel(c, r || 'r1') : null; }).filter(Boolean);
  layout();
  await Promise.all(ps.map(waitBuilt));
  const comp = Q.get('comp');
  if (comp != null && ps.length) { const p = ps.find((x) => x.cfg !== 'src') || ps[0]; state.selComp = Number(comp); state.compPanel = p; await buildPanel(p); }
  if (Q.get('chunks') === '1') {   // V-5 캡처: 첫 R4 칸을 청크 보기로(swap=before 면 교체 전)
    const p = ps.find((x) => m.runs[x.cfg]?.[x.run]?.chunk_view);
    if (p) { await toggleChunks(p); if (Q.get('swap') === 'before') swapChunks(p); }
  }
  setView(Q.get('view') || 'edit');
  document.body.dataset.ready = '1';
}

function waitBuilt(p) {
  return new Promise((res) => { const t = setInterval(() => { if (p.mesh || p.legend.textContent.includes('GLB 실패')) { clearInterval(t); res(); } }, 100); });
}

main();

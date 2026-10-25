// 목록·상세 페이지가 같이 쓰는 three.js 도우미.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export { THREE };

export function fileUrl(id, name) {
  return `/file?id=${encodeURIComponent(id)}&name=${encodeURIComponent(name)}`;
}

const loader = new GLTFLoader();

export function loadGLB(url) {
  return new Promise((resolve, reject) => loader.load(url, (g) => resolve(g.scene), undefined, reject));
}

export function addLights(scene) {
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444450, 1.6));
  const d1 = new THREE.DirectionalLight(0xffffff, 1.6);
  d1.position.set(2, 3, 2);
  scene.add(d1);
  const d2 = new THREE.DirectionalLight(0xffffff, 0.6);
  d2.position.set(-2, 1, -2);
  scene.add(d2);
}

// 정점색이 있는 메시는 정점색을 쓰고, 재질이 없으면 회색 재질을 준다.
export function prepare(obj) {
  let verts = 0, faces = 0;
  obj.traverse((o) => {
    if (!o.isMesh) return;
    const g = o.geometry;
    verts += g.attributes.position ? g.attributes.position.count : 0;
    faces += g.index ? g.index.count / 3 : (g.attributes.position ? g.attributes.position.count / 3 : 0);
    const hasColor = !!g.attributes.color;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    mats.forEach((m) => {
      if (!m) return;
      if (hasColor) m.vertexColors = true;
      m.side = THREE.DoubleSide;
      m.needsUpdate = true;
    });
  });
  return { verts, faces };
}

// 물체 bbox 에 맞춰 카메라 위치를 잡는다. dir 은 보는 방향(정규화 전).
export function frame(camera, obj, target, dir = new THREE.Vector3(1, 0.6, 1.2)) {
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 1e-3);
  const dist = radius / Math.sin((camera.fov * Math.PI) / 360) * 1.05;
  camera.near = dist / 100;
  camera.far = dist * 100;
  camera.position.copy(center).add(dir.clone().normalize().multiplyScalar(dist));
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  if (target) target.copy(center);
  return { box, size, center, radius };
}

// 썸네일 1장 렌더(오프스크린). PNG Blob 을 돌려준다.
let thumbRenderer = null;
export async function renderThumb(url, w = 320, h = 240) {
  if (!thumbRenderer) {
    thumbRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    thumbRenderer.setPixelRatio(1);
  }
  thumbRenderer.setSize(w, h, false);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf2f2f4);
  addLights(scene);
  const obj = await loadGLB(url);
  prepare(obj);
  scene.add(obj);
  const cam = new THREE.PerspectiveCamera(35, w / h, 0.01, 100);
  frame(cam, obj, null);
  thumbRenderer.render(scene, cam);
  // toBlob 은 일부 환경(headless·소프트웨어 WebGL)에서 콜백이 오지 않는다 → toDataURL(동기) 사용.
  const dataUrl = thumbRenderer.domElement.toDataURL('image/png');
  const blob = await (await fetch(dataUrl)).blob();
  obj.traverse((o) => {
    if (o.isMesh) {
      o.geometry.dispose();
      (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => {
        if (!m) return;
        Object.values(m).forEach((v) => v && v.isTexture && v.dispose());
        m.dispose();
      });
    }
  });
  return blob;
}

// ── T1 시나리오 페이지: 좌표 변환·카메라는 이 두 함수에서만 한다 ──────────────────────────────
// 셀 (i,j,k) → TRELLIS 복셀 프레임 ((i,j,k)+0.5)/64 − 0.5 → glTF(Y-up) (x, z, −y) → 시나리오 정규화 (p − c) × s.
// xf = meta.xf {center:[cx,cy,cz](glTF 프레임), scale}. 트랙 B 는 항등(Bench 프레임). scenario_prep.py:cells_to_world 와 같은 식.
export function voxelToWorld(flatCells, xf) {
  const n = flatCells.length / 3, out = new Float32Array(n * 3);
  const [cx, cy, cz] = xf.center, s = xf.scale;
  for (let t = 0; t < n; t++) {
    const x = (flatCells[3 * t] + 0.5) / 64 - 0.5, y = (flatCells[3 * t + 1] + 0.5) / 64 - 0.5, z = (flatCells[3 * t + 2] + 0.5) / 64 - 0.5;
    out[3 * t] = (x - cx) * s; out[3 * t + 1] = (z - cy) * s; out[3 * t + 2] = (-y - cz) * s;
  }
  return out;
}

// 결과 GLB(to_glb 원 프레임)를 같은 정규화로 놓는다: p' = (p − c) × s.
export function placeNormalized(obj, xf) {
  obj.scale.setScalar(xf.scale);
  obj.position.set(-xf.center[0] * xf.scale, -xf.center[1] * xf.scale, -xf.center[2] * xf.scale);
}

// 포크 조건 렌더 카메라(utils/render_rgb_and_mask.py + bpyrenderer 기본값): 반지름 1.8, 고도 15°,
// 방위 [90,45,0,315,270][v] − 90, 원점 주시, 세로 화각 49.13°(센서 32 mm·렌즈 35 mm, 정사각).
// Blender(Z-up) (x,y,z) → three/glTF(Y-up) (x, z, −y).
export const FORK_FOV = 2 * Math.atan(16 / 35) * 180 / Math.PI;
export function forkCamera(camera, view, cam = { radius: 1.8, elev_deg: 15, azimuth_deg: [90, 45, 0, 315, 270], azimuth_offset_deg: -90 }) {
  const el = cam.elev_deg * Math.PI / 180, az = (cam.azimuth_deg[view] + cam.azimuth_offset_deg) * Math.PI / 180;
  const bx = cam.radius * Math.cos(el) * Math.cos(az), by = cam.radius * Math.cos(el) * Math.sin(az), bz = cam.radius * Math.sin(el);
  camera.fov = FORK_FOV;
  camera.position.set(bx, bz, -by);
  camera.up.set(0, 1, 0);
  camera.lookAt(0, 0, 0);
  camera.near = 0.01; camera.far = 100;
  camera.updateProjectionMatrix();
}

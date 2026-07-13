/* ───────────────────────────────────────────────────────────────────────────
 * Fully client-side sign-language recognition.
 *
 * Pipeline (all in-browser, zero server inference):
 *   webcam frame  →  MediaPipe Tasks (Hands + Face + Pose)
 *               →  feature_extractor.js (216-dim, parity-checked vs Python)
 *               →  45-frame buffer  →  onnxruntime-web (CNN-BiLSTM-Attn)
 *               →  prediction overlaid on the video
 *
 * Replaces the old WebSocket app.js that streamed JPEGs to a FastAPI server.
 * ─────────────────────────────────────────────────────────────────────────── */
import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
  PoseLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';
import { SignRecognizer } from './model.js';

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

/* ── State ─────────────────────────────────────────────────────────────── */
let stream        = null;
let cameraRunning = false;
let mirrored      = true;
let showHands     = true;
let showFace      = true;
let busy          = false;
let lastTs        = 0;

let handLandmarker = null;
let faceLandmarker = null;
let poseLandmarker = null;
const recognizer = new SignRecognizer();

// FPS / inference-time tracking
let fpsCounter   = 0;
let fpsTimestamp = performance.now();

// Landmark interpolation (smooth overlay independent of detection rate).
let tgtHands = null, curHands = null;
let tgtFace  = null, curFace  = null;
let lastConnections = null;
const LERP = 0.5;

/* ── DOM refs ──────────────────────────────────────────────────────────── */
const video       = document.getElementById('video');
const overlay     = document.getElementById('overlay');
const ctx         = overlay.getContext('2d');
const bufBar      = document.getElementById('buffer-bar');
const bufText     = document.getElementById('buffer-text');
const predLetter  = document.getElementById('pred-letter');
const predConf    = document.getElementById('pred-confidence');
const predBox     = document.getElementById('prediction-box');
const predLetterMobile = document.getElementById('pred-letter-mobile');
const predConfMobile   = document.getElementById('pred-confidence-mobile');
const predBoxMobile    = document.getElementById('prediction-box-mobile');
const statusBadge = document.getElementById('status-badge');
const statFps     = document.getElementById('stat-fps');
const statRtt     = document.getElementById('stat-rtt');
const statHands   = document.getElementById('stat-hands');
const statMode    = document.getElementById('stat-mode');
const refSubtitle = document.getElementById('ref-subtitle');
const camSubtitle = document.getElementById('cam-subtitle');
const leftSidebar = document.getElementById('left-sidebar');
const rightSidebar = document.getElementById('right-sidebar');
const mobileOverlay = document.getElementById('mobile-overlay');
const btnLeftMenu = document.getElementById('btn-left-menu');
const btnRightMenu = document.getElementById('btn-right-menu');
const btnToggleAlphabet = document.getElementById('btn-toggle-alphabet');
const btnToggleSemne = document.getElementById('btn-toggle-semne');
const alphabetSection = document.getElementById('alphabet-section');
const semneSection = document.getElementById('semne-section');
const semneList = document.getElementById('semne-list');

/* ── Mobile menus ──────────────────────────────────────────────────────── */
let leftMenuOpen = false;
let rightMenuOpen = false;

function isMobileLayout() {
  return window.matchMedia && window.matchMedia('(max-width: 900px)').matches;
}

function syncSidebarLayout() {
  if (!leftSidebar || !rightSidebar) return;
  if (isMobileLayout()) {
    leftSidebar.classList.add('is-mobile', 'left');
    rightSidebar.classList.add('is-mobile', 'right');
    if (!leftMenuOpen && !rightMenuOpen && mobileOverlay) mobileOverlay.classList.add('hidden');
  } else {
    leftSidebar.classList.remove('is-mobile', 'left', 'open');
    rightSidebar.classList.remove('is-mobile', 'right', 'open');
    leftMenuOpen = false;
    rightMenuOpen = false;
    if (mobileOverlay) mobileOverlay.classList.add('hidden');
  }
}

function setMenus({ left, right }) {
  leftMenuOpen = !!left;
  rightMenuOpen = !!right;
  if (!isMobileLayout()) return;
  if (leftSidebar) leftSidebar.classList.toggle('open', leftMenuOpen);
  if (rightSidebar) rightSidebar.classList.toggle('open', rightMenuOpen);
  if (mobileOverlay) mobileOverlay.classList.toggle('hidden', !(leftMenuOpen || rightMenuOpen));
}

/* ── Model + landmarker init ───────────────────────────────────────────── */
async function initEngine() {
  setStatus('loading');
  if (camSubtitle) camSubtitle.textContent = 'Loading models…';
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    [handLandmarker, faceLandmarker, poseLandmarker] = await Promise.all([
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
        numHands: 2, runningMode: 'VIDEO',
        minHandDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
      }),
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
        numFaces: 1, runningMode: 'VIDEO',
      }),
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
        numPoses: 1, runningMode: 'VIDEO',
      }),
    ]);
    await recognizer.load();
    setStatus('online');
    if (camSubtitle) camSubtitle.textContent = 'Camera stopped';
  } catch (err) {
    console.error('Engine init failed:', err);
    setStatus('offline');
    if (camSubtitle) camSubtitle.textContent = 'Model load failed';
    alert(`Could not load models: ${err.message}`);
  }
}

/* ── Camera ────────────────────────────────────────────────────────────── */
async function startCamera() {
  if (cameraRunning) { stopCamera(); return; }
  if (!handLandmarker) { alert('Models are still loading — try again in a moment.'); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    video.addEventListener('loadedmetadata', resizeOverlay, { once: true });
    window.addEventListener('resize', resizeOverlay);
    resizeOverlay();

    cameraRunning = true;
    const btn = document.getElementById('btn-start');
    btn.textContent = 'Stop Camera';
    btn.classList.add('btn-danger');
    btn.classList.remove('btn-primary');
    if (camSubtitle) camSubtitle.textContent = 'Detecting…';
    statMode.textContent = 'live';
  } catch (err) {
    alert(`Camera error: ${err.message}`);
  }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.srcObject = null;
  cameraRunning = false;
  const btn = document.getElementById('btn-start');
  btn.textContent = 'Start Camera';
  btn.classList.add('btn-primary');
  btn.classList.remove('btn-danger');
  if (camSubtitle) camSubtitle.textContent = 'Camera stopped';
  statMode.textContent = 'idle';
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width  = rect.width  || video.videoWidth  || 320;
  overlay.height = rect.height || video.videoHeight || 240;
}

/* ── Per-frame inference loop ──────────────────────────────────────────── */
async function frameTick() {
  if (cameraRunning && !busy && video.readyState >= 2 && video.videoWidth) {
    busy = true;
    const t0 = performance.now();
    const ts = Math.max(lastTs + 1, Math.round(t0));
    lastTs = ts;
    try {
      // MediaPipe reads the raw (non-mirrored) video buffer — CSS mirror on the
      // <video> element does not affect these pixels, so handedness labels match
      // the training pipeline.
      const handRes = handLandmarker.detectForVideo(video, ts);
      const faceRes = faceLandmarker.detectForVideo(video, ts);
      const poseRes = poseLandmarker.detectForVideo(video, ts);

      const hands = handRes.landmarks || [];
      // MediaPipe Tasks reports true anatomical handedness; the model was trained
      // on MediaPipe Solutions labels (which assume a mirrored image), so the same
      // physical hand lands in the opposite slot. Swap Left<->Right to match.
      // Toggle live from the console: window.SWAP_HANDS = false to disable.
      const swap = window.SWAP_HANDS !== false; // default ON
      const handedness = (handRes.handednesses || handRes.handedness || []).map((h) => {
        let label = h[0]?.categoryName;
        if (swap) label = label === 'Left' ? 'Right' : (label === 'Right' ? 'Left' : label);
        return { label, score: h[0]?.score ?? 0 };
      });
      const face = (faceRes.faceLandmarks && faceRes.faceLandmarks[0]) || null;
      const pose = (poseRes.landmarks && poseRes.landmarks[0]) || null;

      const result = await recognizer.process({ hands, handedness, face, pose });

      statRtt.textContent = `${Math.round(performance.now() - t0)} ms`;
      updateBuffer(result.buffer_fill, result.buffer_max);
      tgtHands = result.hands || [];
      tgtFace  = result.face  || [];
      if (result.hand_connections) lastConnections = result.hand_connections;
      updatePrediction(result.prediction, result.confidence, result.top_predictions, result.frozen);
      updateStats(result.hands ? result.hands.length : 0);
      updateFPS();
    } catch (err) {
      console.error('Frame processing error:', err);
    } finally {
      busy = false;
    }
  }
  requestAnimationFrame(frameTick);
}

/* ── Overlay toggles ───────────────────────────────────────────────────── */
function toggleHands() {
  showHands = !showHands;
  const btn = document.getElementById('btn-hands');
  btn.classList.toggle('toggle-on', showHands);
  btn.style.opacity = showHands ? '' : '0.4';
  if (!showHands) ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function toggleFace() {
  showFace = !showFace;
  const btn = document.getElementById('btn-face');
  btn.classList.toggle('toggle-on', showFace);
  btn.style.opacity = showFace ? '' : '0.4';
}

function toggleMirror() {
  mirrored = !mirrored;
  video.style.transform = mirrored ? 'scaleX(-1)' : '';
  document.getElementById('btn-mirror').classList.toggle('mirror-on', mirrored);
}

function resetBuffer() {
  recognizer.reset();
  predLetter.textContent = '—';
  predConf.textContent   = 'waiting for buffer…';
  predBox.classList.remove('has-pred');
  if (predLetterMobile) predLetterMobile.textContent = '—';
  if (predConfMobile) predConfMobile.textContent = 'waiting for buffer…';
  if (predBoxMobile) predBoxMobile.classList.remove('has-pred');
  updateBuffer(0, 45);
}

/* ── UI update helpers ─────────────────────────────────────────────────── */
function setStatus(state) {
  statusBadge.className = `badge badge-${state}`;
  statusBadge.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

function updateBuffer(fill, max) {
  const pct = max ? (fill / max) * 100 : 0;
  bufBar.style.width = `${pct}%`;
  bufText.textContent = `${fill} / ${max}`;
}

function updatePrediction(pred, conf, _top, frozen) {
  if (!pred) return;
  const suffix = frozen ? ' (held — show a hand to resume)' : ' confidence';
  const confText = `${(conf * 100).toFixed(1)} %${suffix}`;
  predLetter.textContent = pred;
  predConf.textContent   = confText;
  predBox.classList.add('has-pred');
  predBox.classList.toggle('is-frozen', !!frozen);
  if (predLetterMobile) predLetterMobile.textContent = pred;
  if (predConfMobile) predConfMobile.textContent = confText;
  if (predBoxMobile) {
    predBoxMobile.classList.add('has-pred');
    predBoxMobile.classList.toggle('is-frozen', !!frozen);
  }
}

function updateStats(handCount) { statHands.textContent = handCount; }

function updateFPS() {
  fpsCounter++;
  const now = performance.now();
  const elapsed = (now - fpsTimestamp) / 1000;
  if (elapsed >= 1) {
    statFps.textContent = Math.round(fpsCounter / elapsed);
    fpsCounter = 0;
    fpsTimestamp = now;
  }
}

/* ── Landmark drawing ──────────────────────────────────────────────────── */
const HAND_COLORS = ['#4f8ef7', '#7c5af7'];
const FACE_COLOR  = '#f7c34f';
const CONN_COLOR  = ['rgba(79,142,247,.6)', 'rgba(124,90,247,.6)'];
const LANDMARK_R  = 4;

function drawLandmarks(hands, face, connections) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!overlay.width) return;
  const W = overlay.width, H = overlay.height;
  const px = (x) => (mirrored ? 1 - x : x) * W;
  const py = (y) => y * H;

  if (showHands && hands) {
    hands.forEach((pts, hi) => {
      if (connections) {
        ctx.strokeStyle = CONN_COLOR[hi] || CONN_COLOR[0];
        ctx.lineWidth = 2;
        connections.forEach(([a, b]) => {
          if (!pts[a] || !pts[b]) return;
          ctx.beginPath();
          ctx.moveTo(px(pts[a][0]), py(pts[a][1]));
          ctx.lineTo(px(pts[b][0]), py(pts[b][1]));
          ctx.stroke();
        });
      }
      pts.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(px(x), py(y), LANDMARK_R, 0, Math.PI * 2);
        ctx.fillStyle = HAND_COLORS[hi] || HAND_COLORS[0];
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
      });
    });
  }

  if (showFace && face && face.length) {
    face.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(px(x), py(y), 3, 0, Math.PI * 2);
      ctx.fillStyle = FACE_COLOR;
      ctx.fill();
    });
  }
}

function _lerpPts(cur, tgt) {
  const out = new Array(tgt.length);
  for (let i = 0; i < tgt.length; i++) {
    const c = cur[i] || tgt[i];
    out[i] = [c[0] + (tgt[i][0] - c[0]) * LERP, c[1] + (tgt[i][1] - c[1]) * LERP];
  }
  return out;
}
function _lerpHands(cur, tgt) {
  if (!tgt) return null;
  if (!cur || cur.length !== tgt.length) return tgt.map((h) => h.map((p) => [p[0], p[1]]));
  return tgt.map((hand, hi) => (cur[hi] && cur[hi].length === hand.length ? _lerpPts(cur[hi], hand) : hand));
}
function _lerpFace(cur, tgt) {
  if (!tgt) return null;
  if (!cur || cur.length !== tgt.length) return tgt.map((p) => [p[0], p[1]]);
  return _lerpPts(cur, tgt);
}
function renderLoop() {
  curHands = _lerpHands(curHands, tgtHands);
  curFace  = _lerpFace(curFace, tgtFace);
  drawLandmarks(curHands, curFace, lastConnections);
  requestAnimationFrame(renderLoop);
}

/* ── Keyboard shortcuts ────────────────────────────────────────────────── */
document.addEventListener('keydown', (e) => {
  switch (e.key.toLowerCase()) {
    case 'r': resetBuffer();  break;
    case 'm': toggleMirror(); break;
    case 's': startCamera();  break;
    case 'h': toggleHands();  break;
    case 'f': toggleFace();   break;
  }
});

/* ── Reference video list ──────────────────────────────────────────────── */
const refVideo       = document.getElementById('ref-video');
const refPlaceholder = document.getElementById('ref-placeholder');
const letterList     = document.getElementById('letter-list');

function selectLetter(item, url) {
  letterList.querySelectorAll('.letter-item').forEach((el) => el.classList.remove('active'));
  if (semneList) semneList.querySelectorAll('.letter-item').forEach((el) => el.classList.remove('active'));
  item.classList.add('active');
  if (refSubtitle) refSubtitle.textContent = `Letter: ${item.textContent}`;
  if (isMobileLayout()) setMenus({ left: false, right: false });
  if (url) {
    refPlaceholder.classList.add('hidden');
    refVideo.classList.remove('hidden');
    refVideo.src = url;
    refVideo.load();
    refVideo.play().catch(() => {});
  } else {
    refVideo.pause();
    refVideo.removeAttribute('src');
    refVideo.load();
    refVideo.classList.add('hidden');
    refPlaceholder.classList.remove('hidden');
    refPlaceholder.textContent = 'No reference video for this item';
  }
}

/* ── Keyboard navigation across the whole left sidebar (WCAG 2.1.1) ─────────
   One continuous arrow-navigable sequence, in DOM order:
     Alphabet header → letters → Semne header → signs.
   ArrowDown / ArrowRight → next, ArrowUp / ArrowLeft → previous,
   Home / End → first / last. Wraps at both ends. Collapsed (hidden) sections
   are skipped automatically. Landing on a sign plays its reference video
   (selection follows focus); landing on a section header just moves focus, so
   Enter/Space can still expand or collapse it. Rebuilt on every keypress, so it
   stays correct as sections open/close and async items load. */
function selectItem(item) {
  selectLetter(item, item.dataset.url || null);
}

// Every element a keyboard user can step onto inside the sidebar.
const SIDEBAR_NAV_SELECTOR = '.sidebar-toggle, .letter-item';

function enableSidebarKeyboardNav(sidebar) {
  if (!sidebar) return;
  sidebar.addEventListener('keydown', (e) => {
    const current = e.target.closest(SIDEBAR_NAV_SELECTOR);
    if (!current || !sidebar.contains(current)) return;

    // Visible items only — offsetParent is null for anything in a hidden section.
    const items = Array.from(sidebar.querySelectorAll(SIDEBAR_NAV_SELECTOR))
      .filter((el) => el.offsetParent !== null);
    const i = items.indexOf(current);
    if (i === -1) return;

    let next;
    switch (e.key) {
      case 'ArrowDown':
      case 'ArrowRight': next = items[(i + 1) % items.length];              break;
      case 'ArrowUp':
      case 'ArrowLeft':  next = items[(i - 1 + items.length) % items.length]; break;
      case 'Home':       next = items[0];                                   break;
      case 'End':        next = items[items.length - 1];                    break;
      default: return;   // let every other key (Tab, Enter, Space…) behave normally
    }

    e.preventDefault();  // stop the arrow keys from scrolling the sidebar
    next.focus();
    // Selection follows focus only for actual signs, not section headers.
    if (next.classList.contains('letter-item')) selectItem(next);
  });
}

async function fetchVideoList() {
  try {
    const res  = await fetch('./assets/videos.json');
    const data = await res.json();
    if (!data.videos || !data.videos.length) return;
    data.videos.forEach(({ letter, url }) => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'letter-item';
      item.textContent = letter;
      item.dataset.url = url || '';
      item.addEventListener('click', () => selectLetter(item, url));
      letterList.appendChild(item);
    });
  } catch (e) {
    console.warn('Could not load video list:', e);
  }
}

/* ── Boot ──────────────────────────────────────────────────────────────── */
video.style.transform = 'scaleX(-1)';
document.getElementById('btn-mirror').classList.add('mirror-on');

function setCollapsed(btn, section, collapsed) {
  if (!btn || !section) return;
  section.classList.toggle('hidden', collapsed);
  btn.setAttribute('aria-expanded', String(!collapsed));
  const chev = btn.querySelector('.chev');
  if (chev) {
    chev.classList.toggle('chev-right', collapsed);
    chev.classList.toggle('chev-down', !collapsed);
  }
}

let alphabetCollapsed = false;
let semneCollapsed = true;
if (btnToggleAlphabet) {
  btnToggleAlphabet.addEventListener('click', () => {
    alphabetCollapsed = !alphabetCollapsed;
    setCollapsed(btnToggleAlphabet, alphabetSection, alphabetCollapsed);
  });
}
if (btnToggleSemne) {
  btnToggleSemne.addEventListener('click', () => {
    semneCollapsed = !semneCollapsed;
    setCollapsed(btnToggleSemne, semneSection, semneCollapsed);
  });
}
setCollapsed(btnToggleAlphabet, alphabetSection, alphabetCollapsed);
setCollapsed(btnToggleSemne, semneSection, semneCollapsed);

if (semneList) {
  ['Hello', 'Thank you', 'Please', 'Yes', 'No', 'Help'].forEach((name) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'letter-item';
    item.textContent = name;
    item.dataset.url = '';
    item.addEventListener('click', () => selectLetter(item, null));
    semneList.appendChild(item);
  });
}

/* Enable arrow-key navigation across the entire left sidebar — section headers
   and both sign lists — as one continuous sequence. Delegated on the container,
   so it also covers the alphabet items loaded asynchronously by fetchVideoList. */
enableSidebarKeyboardNav(document.getElementById('left-sidebar'));

syncSidebarLayout();
window.addEventListener('resize', syncSidebarLayout);
if (btnLeftMenu) btnLeftMenu.addEventListener('click', () => setMenus({ left: !leftMenuOpen, right: false }));
if (btnRightMenu) btnRightMenu.addEventListener('click', () => setMenus({ left: false, right: !rightMenuOpen }));
if (mobileOverlay) mobileOverlay.addEventListener('click', () => setMenus({ left: false, right: false }));

// Expose handlers used by inline onclick attributes in index.html.
Object.assign(window, { startCamera, toggleMirror, resetBuffer, toggleHands, toggleFace });

fetchVideoList();
requestAnimationFrame(renderLoop);
requestAnimationFrame(frameTick);
initEngine();

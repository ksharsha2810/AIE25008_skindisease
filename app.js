'use strict';

const LABELS = {
  actinic_keratosis_or_intraepithelial_carcinoma: 'Actinic keratosis / intraepithelial carcinoma (AKIEC)',
  basal_cell_carcinoma: 'Basal cell carcinoma (BCC)',
  benign_keratosis: 'Benign keratosis (BKL)',
  dermatofibroma: 'Dermatofibroma (DF)',
  melanoma: 'Melanoma (MEL)',
  nevus: 'Melanocytic nevus (NV)',
  vascular_lesion: 'Vascular lesion (VASC)',
  grade_1: 'Grade 1 (mild)', grade_2: 'Grade 2 (moderate)', grade_3: 'Grade 3 (severe)', grade_4: 'Grade 4 (very severe)',
};
const SAMPLES = ['melanoma', 'nevus', 'basal_cell_carcinoma', 'benign_keratosis', 'vascular_lesion'];

let session = null;
let heads = null;
const $ = (id) => document.getElementById(id);

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function linear(head, z, mask) {
  return head.weight.map((row, k) => {
    let s = head.bias[k];
    for (let i = 0; i < z.length; i++) s += row[i] * z[i] * (mask ? mask[i] : 1);
    return s;
  });
}

function normEntropy(p) {
  const h = -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0);
  return Math.min(1, Math.max(0, h / Math.log(p.length)));
}

// Monte Carlo dropout on the pooled vector, matching torch.nn.Dropout (inverted dropout, p = 0.3).
function mcDropout(head, z, passes, p, rng) {
  const keep = 1 - p;
  const mean = new Array(head.bias.length).fill(0);
  for (let t = 0; t < passes; t++) {
    const mask = z.map(() => (rng() < keep ? 1 / keep : 0));
    softmax(linear(head, z, mask)).forEach((v, k) => { mean[k] += v / passes; });
  }
  return mean;
}

function preprocess(image) {
  const size = heads.preprocess.size;
  const scale = size / Math.min(image.width, image.height);
  const w = Math.round(image.width * scale);
  const h = Math.round(image.height * scale);
  const canvas = $('input');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(image, Math.floor((size - w) / 2), Math.floor((size - h) / 2), w, h);
  const { data } = ctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  const { mean, std } = heads.preprocess;
  for (let i = 0; i < size * size; i++) {
    for (let c = 0; c < 3; c++) tensor[c * size * size + i] = (data[i * 4 + c] / 255 - mean[c]) / std[c];
  }
  return new ort.Tensor('float32', tensor, [1, 3, size, size]);
}

function jet(v) {
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)));
  return [r * 255, g * 255, b * 255];
}

function drawCam(refined, classIndex) {
  const [, channels, gh, gw] = refined.dims;
  const weights = heads.lesion.weight[classIndex];
  const cam = new Float32Array(gh * gw);
  for (let k = 0; k < channels; k++) {
    for (let i = 0; i < gh * gw; i++) cam[i] += weights[k] * refined.data[k * gh * gw + i];
  }
  let lo = Infinity; let hi = -Infinity;
  cam.forEach((v, i) => { cam[i] = Math.max(0, v); lo = Math.min(lo, cam[i]); hi = Math.max(hi, cam[i]); });
  const small = document.createElement('canvas');
  small.width = gw; small.height = gh;
  const sctx = small.getContext('2d');
  const img = sctx.createImageData(gw, gh);
  cam.forEach((v, i) => {
    const [r, g, b] = jet(hi > lo ? (v - lo) / (hi - lo) : 0);
    img.data.set([r, g, b, 255], i * 4);
  });
  sctx.putImageData(img, 0, 0);
  const out = $('cam').getContext('2d');
  out.drawImage($('input'), 0, 0);
  out.globalAlpha = 0.45;
  out.imageSmoothingEnabled = true;
  out.drawImage(small, 0, 0, 224, 224);
  out.globalAlpha = 1;
}

function renderTable(tableId, classes, probs) {
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  $(tableId).innerHTML = order.map(([p, i], rank) =>
    `<tr class="${rank === 0 ? 'top' : ''}"><td>${LABELS[classes[i]]}</td><td style="width:35%"><div class="bar" style="width:${(p * 100).toFixed(1)}%"></div></td><td>${(p * 100).toFixed(1)}%</td></tr>`
  ).join('');
}

async function analyse(image) {
  $('status').textContent = 'Running model…';
  const feeds = { image: preprocess(image) };
  const result = await session.run(feeds);
  const z = Array.from(result.pooled.data);
  const rng = mulberry32(42);
  const passes = heads.mc_passes;
  const lesion = mcDropout(heads.lesion, z, passes, heads.dropout, rng);
  const acne = mcDropout(heads.acne, z, passes, heads.dropout, rng);
  const lesionDet = softmax(linear(heads.lesion, z));
  const top = lesion.indexOf(Math.max(...lesion));
  const topAcne = acne.indexOf(Math.max(...acne));
  const hL = normEntropy(lesion);
  const hA = normEntropy(acne);
  const flagged = hL > heads.referral_entropy_threshold;

  renderTable('lesionTable', heads.lesion.classes, lesion);
  renderTable('acneTable', heads.acne.classes, acne);
  $('lesionMeta').textContent = `MC-dropout mean of ${passes} passes. Normalised predictive entropy ${hL.toFixed(4)}; deterministic top-1 probability ${(Math.max(...lesionDet) * 100).toFixed(1)}%.`;
  $('acneMeta').textContent = `Normalised predictive entropy ${hA.toFixed(4)}.`;
  $('referral').innerHTML = flagged
    ? `<div class="flag yes">Specialist review recommended: entropy ${hL.toFixed(3)} &gt; ${heads.referral_entropy_threshold}</div>`
    : `<div class="flag no">Below referral threshold (entropy ${hL.toFixed(3)} ≤ ${heads.referral_entropy_threshold})</div>`;
  drawCam(result.refined, top);

  const lines = [
    `Model disease class: ${heads.lesion.classes[top]}. This is a research prediction, not a clinician diagnosis.`,
    `Disease MC-dropout confidence: ${lesion[top].toFixed(4)}; normalized predictive entropy: ${hL.toFixed(4)}.`,
    `ACNE04 head predicted acne grade: ${topAcne + 1}; applicability: false. Not clinically applicable: acne has not been externally confirmed.`,
    `Severity MC-dropout confidence: ${acne[topAcne].toFixed(4)}; normalized predictive entropy: ${hA.toFixed(4)}.`,
    'Class activation map shows image regions weighted by the predicted class; it is not a lesion mask or proof of clinical correctness.',
    'Specialist review required.',
  ];
  $('report').innerHTML = lines.map((l) => `<li>${l}</li>`).join('');
  $('output').hidden = false;
  $('status').textContent = 'Done.';
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

async function init() {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  try {
    [session, heads] = await Promise.all([
      ort.InferenceSession.create('model/lcnet_cbam.onnx', { executionProviders: ['wasm'] }),
      fetch('model/heads.json').then((r) => r.json()),
    ]);
  } catch (error) {
    $('status').textContent = `Could not load the model: ${error.message}`;
    return;
  }
  $('status').textContent = 'Model ready. Upload an image or pick a sample.';
  $('samples').innerHTML = SAMPLES.map((s) =>
    `<button data-src="samples/${s}.jpg"><img src="samples/${s}.jpg" alt="${s}">${LABELS[s].replace(/ \(.*/, '')}</button>`
  ).join('');
  document.querySelectorAll('.samples button').forEach((b) =>
    b.addEventListener('click', async () => analyse(await loadImage(b.dataset.src))));
  $('file').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    try { await analyse(await loadImage(url)); } finally { URL.revokeObjectURL(url); }
  });
}

init();

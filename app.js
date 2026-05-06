// そらまる くずし字認識: ブラウザ完結 ConvNeXt-tiny @ 384 (3673 cls) ONNX Runtime Web 推論
// 必要な静的ファイル: ./model/convnext_v4.{onnx,meta.json}, ./model/unicode_translation.csv

const ORT_VERSION = "1.19.2";
const TOP_K       = 20;
const UNI_URL     = "./model/unicode_translation.csv";

// 重量の大きい .onnx は Hugging Face Hub にホスト（GitHub の 100MB 制限を回避）。
// メタ JSON / Unicode CSV はリポジトリ同梱で十分小さい。
const MODEL_ONNX_URL = "https://huggingface.co/yuta1984/soramaru_kuzushiji_ai/resolve/main/convnext_v4.onnx";

const MODEL = {
  label:       "ConvNeXt-tiny @ 384 (3673 cls)",
  onnxUrl:     MODEL_ONNX_URL,
  metaUrl:     "./model/convnext_v4.meta.json",
  previewEl:   "preview",
  previewMeta: "preview-meta",
  resultsEl:   "results",
  session: null,
  meta: null,
  ready: false,
};

const $ = (id) => document.getElementById(id);
const $statusShared = $("status-shared");
const $dropzone     = $("dropzone");
const $file         = $("file");
const $soramaru     = $("soramaru");
const $bubble       = $("bubble");

// top-1 確率に応じて表情とセリフを切り替え（高いほど自信あり）。
const RESULT_BINS = [
  [0.90, "./soramaru/03_star.png",     "これだ！ 自信あるよ！"],
  [0.70, "./soramaru/08_fun.png",      "こう読んでみたよ。どうかな？"],
  [0.50, "./soramaru/12_smile.png",    "たぶんこれかな？ 他の候補もチェックしてね"],
  [0.30, "./soramaru/19_soramaru.png", "うーん、ちょっと自信ないなあ…"],
  [0.15, "./soramaru/21_soramaru.png", "むずかしい字だね…他の候補も見てね"],
];
const RESULT_FALLBACK = ["./soramaru/22_soramaru.png", "ごめん、ちょっと自信ないかも…"];

function pickResult(prob) {
  for (const [threshold, src, text] of RESULT_BINS) {
    if (prob >= threshold) return [src, text];
  }
  return RESULT_FALLBACK;
}
function setResultMascot(prob) {
  const [src, text] = pickResult(prob);
  $soramaru.src = src;
  $bubble.textContent = text;
}

function setThinkingMascot() {
  $soramaru.src = "./soramaru/11_analyse.png";
  $bubble.textContent = "どれどれ…ふむふむ…";
}

let uniMap = new Map();

function setShared(msg, isError = false) {
  $statusShared.textContent = msg;
  $statusShared.classList.toggle("error", isError);
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.text();
}
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.json();
}

function parseUnicodeCsv(text) {
  const m = new Map();
  const lines = text.split(/\r?\n/);
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    const idx = line.indexOf(",");
    if (idx < 0) continue;
    const code = line.slice(0, idx).trim();
    const ch   = line.slice(idx + 1);
    if (code) m.set(code, ch);
  }
  return m;
}

async function loadModel(m) {
  setShared("モデル情報を読み込み中...");
  m.meta = await fetchJSON(m.metaUrl);
  if (!m.meta.classes || !m.meta.mean || !m.meta.std || !m.meta.input_size) {
    throw new Error("meta.json に classes/mean/std/input_size が無い");
  }
  setShared("モデルを読み込み中...");
  m.session = await ort.InferenceSession.create(m.onnxUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const cv = $(m.previewEl);
  cv.width = m.meta.input_size;
  cv.height = m.meta.input_size;
  m.ready = true;
}

async function init() {
  try {
    setShared("初期化中...");
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    uniMap = parseUnicodeCsv(await fetchText(UNI_URL));
    await loadModel(MODEL);
    setShared("準備完了。画像を貼り付けるか選択してください。");
  } catch (e) {
    console.error(e);
    setShared(`初期化失敗: ${e.message}`, true);
  }
}

function preprocess(bitmap, m) {
  const size = m.meta.input_size;
  const W = bitmap.width, H = bitmap.height;
  const s = Math.min(W, H);
  const sx = (W - s) / 2;
  const sy = (H - s) / 2;
  const cv = $(m.previewEl);
  cv.width = size; cv.height = size;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, sx, sy, s, s, 0, 0, size, size);

  const px = ctx.getImageData(0, 0, size, size).data;
  const N = size * size;
  const buf = new Float32Array(3 * N);
  const mn = m.meta.mean, sd = m.meta.std;
  for (let i = 0, j = 0; i < N; i++, j += 4) {
    buf[i        ] = (px[j    ] / 255 - mn[0]) / sd[0];
    buf[i +   N  ] = (px[j + 1] / 255 - mn[1]) / sd[1];
    buf[i + 2*N  ] = (px[j + 2] / 255 - mn[2]) / sd[2];
  }
  return new ort.Tensor("float32", buf, [1, 3, size, size]);
}

function softmax(arr) {
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
  let sum = 0;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) { const e = Math.exp(arr[i] - max); out[i] = e; sum += e; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}
function topK(scores, k) {
  const idx = new Array(scores.length);
  for (let i = 0; i < scores.length; i++) idx[i] = i;
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k);
}
function codeToGlyph(code) {
  const m = /^U\+([0-9A-Fa-f]+)$/.exec(code);
  if (!m) return null;
  try { return String.fromCodePoint(parseInt(m[1], 16)); } catch { return null; }
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function renderResults(m, top, probs, logits) {
  const max = probs[top[0]] || 1;
  let html = '<table class="results">';
  html += '<thead><tr><th>#</th><th>文字</th><th>コード</th><th>確率</th><th></th></tr></thead><tbody>';
  for (let i = 0; i < top.length; i++) {
    const cls   = top[i];
    const code  = m.meta.classes[cls];
    const glyph = uniMap.get(code) || codeToGlyph(code) || "?";
    const p     = probs[cls];
    const logit = logits[cls];
    const pct   = (p * 100).toFixed(2);
    const w     = Math.max(1, Math.round((p / max) * 100));
    const rowCls = (i === 0) ? ' class="top1"' : "";
    html += `<tr${rowCls} title="logit=${logit.toFixed(3)}">`
         +  `<td class="rank">${i + 1}</td>`
         +  `<td class="glyph">${escapeHtml(glyph)}</td>`
         +  `<td class="code">${code}</td>`
         +  `<td class="prob">${pct}%</td>`
         +  `<td class="bar"><div class="bar-track"><div class="bar-fill" style="width:${w}%"></div></div></td>`
         +  `</tr>`;
  }
  html += "</tbody></table>";
  $(m.resultsEl).innerHTML = html;
}

async function runModel(m, bitmap) {
  const t0 = performance.now();
  const tensor = preprocess(bitmap, m);
  const t1 = performance.now();
  const out = await m.session.run({ [m.session.inputNames[0]]: tensor });
  const logits = out[m.session.outputNames[0]].data;
  const t2 = performance.now();
  const probs = softmax(logits);
  const top   = topK(probs, TOP_K);
  return {
    top, probs, logits,
    pre_ms: t1 - t0,
    inf_ms: t2 - t1,
    total_ms: t2 - t0,
    bw: bitmap.width, bh: bitmap.height,
  };
}

async function handleFile(blob, srcLabel = "input") {
  if (!MODEL.ready) {
    setShared("初期化未完了です。少し待ってから再試行してください。", true);
    return;
  }
  if (!blob || !/^image\//.test(blob.type)) return;

  let bitmap;
  try {
    bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  } catch (e) {
    setShared(`画像のデコード失敗 (${blob.type}): ${e.message}`, true);
    return;
  }

  setShared(`${srcLabel}: ${bitmap.width}×${bitmap.height} を認識中...`);
  setThinkingMascot();
  // wasm 推論はメインスレッドをブロックするため、1 フレーム待って「考え中」表情を描画する
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

  let r;
  try {
    r = await runModel(MODEL, bitmap);
  } catch (e) {
    setShared(`認識失敗: ${e.message}`, true);
    bitmap.close?.();
    return;
  }

  renderResults(MODEL, r.top, r.probs, r.logits);

  $(MODEL.previewMeta).innerHTML =
    `${srcLabel}: ${r.bw}×${r.bh} → ${MODEL.meta.input_size}×${MODEL.meta.input_size}<br>`
    + `preprocess ${r.pre_ms.toFixed(0)} ms / infer <strong>${r.inf_ms.toFixed(0)} ms</strong>`;

  const code  = MODEL.meta.classes[r.top[0]];
  const glyph = uniMap.get(code) || codeToGlyph(code) || "?";
  const pct   = (r.probs[r.top[0]] * 100).toFixed(1);
  setShared(`認識結果: ${glyph} (${code}) ${pct}% / ${r.total_ms.toFixed(0)} ms`);

  setResultMascot(r.probs[r.top[0]]);

  bitmap.close?.();
}

// label が内包 input にクリックを伝えるネイティブ動作に任せる（明示 click 呼び出しは二重ダイアログの原因になる）
$dropzone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    $file.click();
  }
});
$file.addEventListener("change", e => {
  const f = e.target.files && e.target.files[0];
  if (f) handleFile(f, f.name);
  e.target.value = "";
});

document.addEventListener("paste", e => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const it of items) {
    if (it.kind === "file" && /^image\//.test(it.type)) {
      const blob = it.getAsFile();
      if (blob) handleFile(blob, "clipboard");
      e.preventDefault();
      return;
    }
  }
});

init();

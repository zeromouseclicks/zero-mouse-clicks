/* Zero Mouse Clicks — motor en el navegador (Web Worker).
   Pyodide corre el paquete zmc (pipeline v5); onnxruntime-web corre los tres modelos.
   La lectura de textos (tesseract.js) la hace la página principal: se le pide por mensajes.
   Nada sale del equipo del usuario: sin servidor, sin tokens. */
var CDN = {
  pyodide: 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/',
  ort: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'
};
var BASE = self.location.href.replace(/[^/]*$/, '');          // .../app/
var VERSION = 'v4';
importScripts(CDN.pyodide + 'pyodide.js', CDN.ort + 'ort.wasm.min.js');

ort.env.wasm.wasmPaths = CDN.ort;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
ort.env.logLevel = 'error';

var MODELS = { v2: 'models/unet_v2.onnx', v3: 'models/unet_v3.onnx', v4: 'models/unet_v4.onnx' };
var sessions = {}, py = null, ready = null, glue = null, waiting = {}, nextId = 1;

function post(m, t) { self.postMessage(m, t || []); }
function stage(s, f) { post({ type: 'progress', stage: s, frac: f }); }

function bytes(path) {
  return fetch(BASE + path + '?' + VERSION).then(function (r) {
    if (!r.ok) throw new Error('fetch ' + path + ' ' + r.status);
    return r.arrayBuffer();
  });
}

// lean: teléfonos (sobre todo iPhone, con poca memoria por pestaña). Todo se carga de uno en uno
// y los modelos no guardan memoria entre corridas.
var lean = false, step = 'start';

function session(k) {
  var opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  if (lean) { opt.enableCpuMemArena = false; opt.enableMemPattern = false; }
  return bytes(MODELS[k]).then(function (b) { return ort.InferenceSession.create(b, opt); }).then(function (s) { sessions[k] = s; });
}

function boot() {
  if (ready) return ready;
  ready = (async function () {
    stage('boot', 0.02);
    var keys = Object.keys(MODELS), sess = lean ? null : Promise.all(keys.map(session));
    if (sess) sess.catch(function () {});
    step = 'python';
    py = await loadPyodide({ indexURL: CDN.pyodide });
    stage('boot', 0.25);
    step = 'packages';
    if (lean) { await py.loadPackage('numpy'); stage('boot', 0.35); await py.loadPackage('opencv-python'); stage('boot', 0.55); await py.loadPackage(['pillow', 'charset-normalizer']); }
    else await py.loadPackage(['numpy', 'opencv-python', 'pillow', 'charset-normalizer']);
    stage('boot', 0.7);
    step = 'files';
    var files = await Promise.all([bytes('py/zmc.zip'), bytes('py/reportlab.whl'),
      bytes('fonts/DejaVuSans.ttf'), bytes('fonts/DejaVuSans-Bold.ttf')]);
    py.unpackArchive(files[0], 'zip');
    py.unpackArchive(files[1], 'wheel', { extractDir: py.runPython('import sysconfig; sysconfig.get_paths()["purelib"]') });
    py.FS.mkdirTree('/usr/share/fonts/truetype/dejavu');
    py.FS.writeFile('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', new Uint8Array(files[2]));
    py.FS.writeFile('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', new Uint8Array(files[3]));
    files = null;
    step = 'import';
    glue = py.pyimport('zmc_web');
    py.runPython('import zmc.drawing, zmc.sections, zmc.finish, zmc.vectorize5');
    stage('boot', 0.85);
    step = 'models';
    if (sess) await sess; else for (var i = 0; i < keys.length; i++) await session(keys[i]);
    step = 'ready';
    stage('boot', 1);
  })();
  ready.catch(function () { ready = null; });
  return ready;
}

function detail(err) { return 'step ' + step + ' · ' + String(err && (err.stack || err.message) || err).slice(0, 500); }

// avance: fracción al empezar cada etapa y cuántas corridas de modelo se esperan dentro de ella
var STAGES = { area: 0.17, locate: 0.25, read: 0.38, texts: 0.8, vector: 0.93, draw: 0.96 };
var NEXT = { locate: 0.38, read: 0.8 }, RUNS = { locate: 3, read: 12 }, cur = { stage: 'boot', frac: 0 };

async function infer(name, u8, n) {
  var s = sessions[name], f = new Float32Array(u8.length);
  for (var i = 0; i < u8.length; i++) f[i] = u8[i];
  var feeds = {}; feeds[s.inputNames[0]] = new ort.Tensor('float32', f, [1, n, n, 3]);
  var out = await s.run(feeds);
  if (NEXT[cur.stage]) {
    cur.frac = Math.min(NEXT[cur.stage] - 0.01, cur.frac + (NEXT[cur.stage] - STAGES[cur.stage]) / RUNS[cur.stage]);
    stage(cur.stage, cur.frac);
  }
  return out[s.outputNames[0]].data;
}

function ocr(pngs) {
  var id = nextId++, list = [];
  for (var i = 0; i < pngs.length; i++) list.push(pngs[i].slice().buffer);
  return new Promise(function (res, rej) {
    waiting[id] = { res: res, rej: rej };
    post({ type: 'ocr', id: id, pngs: list }, list);
  });
}

self.onmessage = async function (e) {
  var m = e.data;
  if (m.lean) lean = true;
  if (m.type === 'boot') { boot().catch(function (err) { post({ type: 'error', code: 'boot', message: detail(err) }); }); return; }
  if (m.type === 'ocr-result') {
    var w = waiting[m.id]; delete waiting[m.id];
    if (w) { if (m.error) w.rej(new Error(m.error)); else w.res(m.rows); }
    return;
  }
  if (m.type !== 'run') return;
  try { await boot(); } catch (err) { post({ type: 'error', code: 'boot', message: detail(err) }); return; }
  try {
    step = 'run';
    var js = {
      infer: infer, ocr: ocr,
      progress: function (s) { cur = { stage: s, frac: STAGES[s] || 0.5 }; stage(s, cur.frac); },
      preview: function (kind, jpg) { var b = jpg.slice().buffer; post({ type: 'preview', kind: kind, jpg: b }, [b]); }
    };
    var area = (m.area === null || m.area === undefined) ? null : m.area;
    var r = await glue.run(new Uint8Array(m.image), area, m.project || 'Marketing floor plan', m.unit || 'Unit', js, m.areaKind || null, lean);
    var pdf = r.get(0), summary = JSON.parse(r.get(1));
    var copy = pdf.slice();
    r.destroy();
    stage('done', 1);
    post({ type: 'done', pdf: copy.buffer, summary: summary }, [copy.buffer]);
  } catch (err) {
    var s = String(err && err.message || err);
    var code = /NoArea/.test(s) ? 'no_area' : /bad_image/.test(s) ? 'bad_image' : 'failed';
    post({ type: 'error', code: code, message: 'stage ' + cur.stage + ' · ' + s.slice(-600) });
  }
};

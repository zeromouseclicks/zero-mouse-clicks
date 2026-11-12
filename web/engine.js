/* Zero Mouse Clicks — motor en el navegador (Web Worker).
   Pyodide corre el paquete zmc (pipeline v5); onnxruntime-web corre los tres modelos.
   La lectura de textos (tesseract.js) la hace la página principal: se le pide por mensajes.
   Nada sale del equipo del usuario: sin servidor, sin tokens. */
var CDN = {
  pyodide: 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/',
  ort: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'
};
var BASE = self.location.href.replace(/[^/]*$/, '');          // .../app/
var VERSION = 'v1';
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

function boot() {
  if (ready) return ready;
  ready = (async function () {
    stage('boot', 0.02);
    var sess = Object.keys(MODELS).map(function (k) {
      return bytes(MODELS[k]).then(function (b) {
        return ort.InferenceSession.create(b, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      }).then(function (s) { sessions[k] = s; });
    });
    py = await loadPyodide({ indexURL: CDN.pyodide });
    stage('boot', 0.25);
    await py.loadPackage(['numpy', 'opencv-python', 'pillow', 'charset-normalizer']);
    stage('boot', 0.7);
    var files = await Promise.all([bytes('py/zmc.zip'), bytes('py/reportlab.whl'),
      bytes('fonts/DejaVuSans.ttf'), bytes('fonts/DejaVuSans-Bold.ttf')]);
    py.unpackArchive(files[0], 'zip');
    py.unpackArchive(files[1], 'wheel', { extractDir: py.runPython('import sysconfig; sysconfig.get_paths()["purelib"]') });
    py.FS.mkdirTree('/usr/share/fonts/truetype/dejavu');
    py.FS.writeFile('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', new Uint8Array(files[2]));
    py.FS.writeFile('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', new Uint8Array(files[3]));
    glue = py.pyimport('zmc_web');
    py.runPython('import zmc.drawing, zmc.sections, zmc.finish, zmc.vectorize5');
    await Promise.all(sess);
    stage('boot', 1);
  })();
  ready.catch(function () { ready = null; });
  return ready;
}

async function infer(name, u8, n) {
  var s = sessions[name], f = new Float32Array(u8.length);
  for (var i = 0; i < u8.length; i++) f[i] = u8[i];
  var feeds = {}; feeds[s.inputNames[0]] = new ort.Tensor('float32', f, [1, n, n, 3]);
  var out = await s.run(feeds);
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

var STAGES = { area: 0.12, locate: 0.2, read: 0.32, texts: 0.62, vector: 0.82, draw: 0.92 };

self.onmessage = async function (e) {
  var m = e.data;
  if (m.type === 'boot') { boot().catch(function (err) { post({ type: 'error', code: 'boot', message: String(err) }); }); return; }
  if (m.type === 'ocr-result') {
    var w = waiting[m.id]; delete waiting[m.id];
    if (w) { if (m.error) w.rej(new Error(m.error)); else w.res(m.rows); }
    return;
  }
  if (m.type !== 'run') return;
  try {
    await boot();
    var js = {
      infer: infer, ocr: ocr,
      progress: function (s) { stage(s, STAGES[s] || 0.5); }
    };
    var area = (m.area === null || m.area === undefined) ? null : m.area;
    var r = await glue.run(new Uint8Array(m.image), area, m.project || 'Marketing floor plan', m.unit || 'Unit', js);
    var pdf = r.get(0), summary = JSON.parse(r.get(1));
    var copy = pdf.slice();
    r.destroy();
    stage('done', 1);
    post({ type: 'done', pdf: copy.buffer, summary: summary }, [copy.buffer]);
  } catch (err) {
    var s = String(err && err.message || err);
    var code = /NoArea/.test(s) ? 'no_area' : /bad_image/.test(s) ? 'bad_image' : 'failed';
    post({ type: 'error', code: code, message: s.slice(-600) });
  }
};

/* Zero Mouse Clicks — cliente del motor en el navegador.
   ZMC.warm()                         empieza a descargar el motor (al elegir la imagen)
   ZMC.generate(file, area, onStage, onPreview)  -> Promise<{ pdf: Blob, summary }>
       onStage(stage, fraction)   boot, area, locate, read, texts, vector, draw, done
       onPreview(kind, blob)      found, seen, drawn: JPG of what the engine has so far
   Errores: err.code = 'no_area' | 'bad_image' | 'boot' | 'failed' */
(function () {
  var TESS = {
    js: 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js',
    worker: 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/worker.min.js',
    core: 'https://cdn.jsdelivr.net/npm/tesseract.js-core@5.1.1',
    lang: 'https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng/4.0.0_best_int'
  };
  var here = (document.currentScript && document.currentScript.src || location.href).replace(/[^/]*$/, '');
  var worker = null, sched = null, job = null;

  function load(src) {
    return new Promise(function (res, rej) {
      var s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = function () { rej(new Error('load ' + src)); };
      document.head.appendChild(s);
    });
  }

  function ocrReady() {
    if (sched) return sched;
    sched = load(TESS.js).then(async function () {
      var s = Tesseract.createScheduler();
      var n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
      if (/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)) n = Math.min(n, 2);
      var ws = [];
      for (var i = 0; i < n; i++) {
        ws.push(Tesseract.createWorker('eng', 1, { workerPath: TESS.worker, corePath: TESS.core, langPath: TESS.lang })
          .then(function (w) { return w.setParameters({ tessedit_pageseg_mode: '11' }).then(function () { s.addWorker(w); }); }));
      }
      await Promise.all(ws);
      return s;
    });
    sched.catch(function () { sched = null; });
    return sched;
  }

  function ocr(pngs) {
    return ocrReady().then(function (s) {
      return Promise.all(pngs.map(function (b) {
        return s.addJob('recognize', new Blob([b], { type: 'image/png' })).then(function (r) {
          return (r.data.words || []).filter(function (w) { return w.text.trim() && w.confidence >= 40; }).map(function (w) {
            return [w.text.trim(), w.confidence, w.bbox.x0, w.bbox.y0, w.bbox.x1 - w.bbox.x0, w.bbox.y1 - w.bbox.y0];
          });
        });
      }));
    });
  }

  function engine() {
    if (worker) return worker;
    worker = new Worker(here + 'engine.js?v2');
    worker.onmessage = function (e) {
      var m = e.data;
      if (m.type === 'ocr') {
        ocr(m.pngs).then(function (rows) { worker.postMessage({ type: 'ocr-result', id: m.id, rows: rows }); },
          function (err) { worker.postMessage({ type: 'ocr-result', id: m.id, error: String(err) }); });
        return;
      }
      if (!job) return;
      if (m.type === 'progress') job.stage(m.stage, m.frac);
      else if (m.type === 'preview') job.preview(m.kind, new Blob([m.jpg], { type: 'image/jpeg' }));
      else if (m.type === 'done') { var j = job; job = null; j.res({ pdf: new Blob([m.pdf], { type: 'application/pdf' }), summary: m.summary }); }
      else if (m.type === 'error') {
        var k = job; job = null;
        var err = new Error(m.message); err.code = m.code;
        if (m.code === 'boot') { worker.terminate(); worker = null; }
        k.rej(err);
      }
    };
    worker.onerror = function (e) {
      if (job) { var k = job; job = null; var err = new Error(e.message || 'worker'); err.code = 'boot'; k.rej(err); }
      worker = null;
    };
    worker.postMessage({ type: 'boot' });
    return worker;
  }

  // la imagen como JPG/PNG derecha (orientación EXIF aplicada) y de máximo 3200 px de lado
  async function prepare(file) {
    var MAX = 3200, bmp;
    try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
    catch (e) {
      bmp = await new Promise(function (res, rej) {
        var u = URL.createObjectURL(file), im = new Image();
        im.onload = function () { res(im); }; im.onerror = function () { var x = new Error('bad_image'); x.code = 'bad_image'; rej(x); };
        im.src = u;
      });
    }
    var w = bmp.width || bmp.naturalWidth, h = bmp.height || bmp.naturalHeight;
    if (!w || !h) { var x = new Error('bad_image'); x.code = 'bad_image'; throw x; }
    if (/^image\/(jpeg|png)$/.test(file.type) && Math.max(w, h) <= MAX) return file.arrayBuffer();
    var r = Math.min(1, MAX / Math.max(w, h)), c = document.createElement('canvas');
    c.width = Math.round(w * r); c.height = Math.round(h * r);
    var g = c.getContext('2d'); g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height);
    g.imageSmoothingQuality = 'high'; g.drawImage(bmp, 0, 0, c.width, c.height);
    var blob = await new Promise(function (res) { c.toBlob(res, 'image/jpeg', 0.95); });
    return blob.arrayBuffer();
  }

  window.ZMC = {
    warm: function () { engine(); ocrReady(); },
    generate: async function (file, area, onStage, onPreview) {
      if (job) throw Object.assign(new Error('busy'), { code: 'busy' });
      var buf = await prepare(file);
      var w = engine(); ocrReady();
      return new Promise(function (res, rej) {
        job = { res: res, rej: rej, stage: onStage || function () {}, preview: onPreview || function () {} };
        w.postMessage({ type: 'run', image: buf, area: area || null }, [buf]);
      });
    }
  };
})();

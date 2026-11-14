/* Zero Mouse Clicks — cliente del motor en el navegador.
   ZMC.warm()                         empieza a descargar el motor (al elegir la imagen)
   ZMC.generate(file, area, onStage, onPreview)  -> Promise<{ pdf: Blob, summary }>
       onStage(stage, fraction)   boot, area, locate, read, texts, vector, draw, done
       onPreview(kind, blob)      found, seen, drawn: JPG of what the engine has so far
   Errores: err.code = 'no_area' | 'bad_image' | 'old' | 'boot' | 'failed'; err.message trae el detalle técnico */
(function () {
  var TESS = {
    js: 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js',
    worker: 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/worker.min.js',
    core: 'https://cdn.jsdelivr.net/npm/tesseract.js-core@5.1.1',
    lang: 'https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng/4.0.0_best_int'
  };
  var here = (document.currentScript && document.currentScript.src || location.href).replace(/[^/]*$/, '');
  var worker = null, sched = null, job = null, up = false, heard = 0, where = '';
  // si el motor se queda mudo mientras carga (red cortada, memoria), se avisa en vez de esperar para siempre
  setInterval(function () {
    if (!job || up || !worker || Date.now() - heard < 120000) return;
    var k = job; job = null;
    try { worker.terminate(); } catch (x) {}
    worker = null;
    k.rej(fail('boot', 'no answer from the engine for 120 s · last: ' + (where || 'nothing')));
  }, 5000);
  var ua = navigator.userAgent;
  var ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  // lean: teléfonos y equipos con poca memoria: una cosa a la vez (ver engine.js)
  var lean = ios || /Mobi|Android/i.test(ua) || (navigator.deviceMemory && navigator.deviceMemory <= 4);

  function fail(code, message) { var e = new Error(message); e.code = code; return e; }
  // WebAssembly con SIMD: Chrome 91+, Firefox 89+, Safari / iOS 16.4+
  function supported() {
    try {
      return typeof WebAssembly === 'object' && WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
    } catch (e) { return false; }
  }

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
      if (lean) n = ios ? 1 : Math.min(n, 2);
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
    worker = new Worker(here + 'engine.js?v5');
    worker.onmessage = function (e) {
      var m = e.data; heard = Date.now(); if (m.type === 'progress') where = m.stage + ' ' + Math.round(m.frac * 100) + '%';
      if (m.type === 'ocr') {
        ocr(m.pngs).then(function (rows) { worker.postMessage({ type: 'ocr-result', id: m.id, rows: rows }); },
          function (err) { worker.postMessage({ type: 'ocr-result', id: m.id, error: String(err) }); });
        return;
      }
      if (!job) return;
      if (m.type === 'progress') {
        if (m.stage !== 'boot' || m.frac >= 1) up = true;
        // phones: the text reader is done after 'texts'; let its workers go before the sheets are drawn (iPhone kills tabs that use too much memory)
        if (lean && sched && (m.stage === 'vector' || m.stage === 'draw')) { var o = sched; sched = null; o.then(function (s) { return s.terminate(); }).catch(function () {}); }
        job.stage(m.stage, m.frac);
      }
      else if (m.type === 'preview') job.preview(m.kind, new Blob([m.jpg], { type: 'image/jpeg' }));
      else if (m.type === 'done') { var j = job; job = null; j.res({ pdf: new Blob([m.pdf], { type: 'application/pdf' }), summary: m.summary }); }
      else if (m.type === 'error') {
        var k = job; job = null;
        if (m.code === 'boot') { worker.terminate(); worker = null; up = false; }
        k.rej(fail(m.code, m.message));
      }
    };
    worker.onerror = function (e) {
      var d = 'worker · ' + (e.message || 'error') + (e.filename ? ' · ' + String(e.filename).split('/').pop() + ':' + e.lineno : '');
      if (job) { var k = job; job = null; k.rej(fail(up ? 'failed' : 'boot', d)); }
      try { worker.terminate(); } catch (x) {}
      worker = null; up = false;
    };
    heard = Date.now(); where = '';
    worker.postMessage({ type: 'boot', lean: lean });
    return worker;
  }

  // la imagen como JPG/PNG derecha (orientación EXIF aplicada) y de máximo 3200 px de lado
  async function prepare(file) {
    var MAX = lean ? 2400 : 3200, bmp;
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
    lean: lean,
    warm: function () { if (!supported()) return; engine(); if (!lean) ocrReady(); },
    generate: async function (file, area, onStage, onPreview, areaKind) {
      if (job) throw fail('busy', 'busy');
      if (!supported()) throw fail('old', 'no WebAssembly SIMD');
      var buf = await prepare(file);
      var w = engine(); if (!lean) ocrReady();
      return new Promise(function (res, rej) {
        job = { res: res, rej: rej, stage: onStage || function () {}, preview: onPreview || function () {} };
        w.postMessage({ type: 'run', image: buf, area: area || null, areaKind: areaKind || null, lean: lean }, [buf]);
      });
    }
  };
})();

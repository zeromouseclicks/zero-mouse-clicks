"""Arma docs/app/ (lo que la página carga para correr el pipeline en el navegador).

    python web/build.py

Copia el paquete zmc (sin modelos) y web/zmc_web.py en docs/app/py/zmc.zip, los modelos ONNX,
el motor (web/engine.js), las fuentes DejaVu y la rueda de reportlab (pura Python).
Volver a correrlo cada vez que cambie el código de zmc/.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, 'docs', 'app')
REPORTLAB = 'reportlab==4.4.4'


def main():
    for d in ('py', 'models', 'fonts'):
        os.makedirs(os.path.join(APP, d), exist_ok=True)
    with zipfile.ZipFile(os.path.join(APP, 'py', 'zmc.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(os.path.join(ROOT, 'zmc', '*.py'))):
            z.write(f, 'zmc/' + os.path.basename(f))
        z.write(os.path.join(ROOT, 'web', 'zmc_web.py'), 'zmc_web.py')
    for f in glob.glob(os.path.join(ROOT, 'zmc', 'models', '*.onnx')):
        shutil.copy(f, os.path.join(APP, 'models'))
    for f in ('engine.js', 'client.js'):
        shutil.copy(os.path.join(ROOT, 'web', f), APP)
    fonts = os.path.join(ROOT, 'web', 'fonts')
    for f in ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'LICENSE-DejaVu.txt'):
        shutil.copy(os.path.join(fonts, f), os.path.join(APP, 'fonts'))
    whl = os.path.join(APP, 'py', 'reportlab.whl')
    if not os.path.exists(whl):
        with tempfile.TemporaryDirectory() as t:
            subprocess.check_call([sys.executable, '-m', 'pip', 'download', REPORTLAB, '--no-deps', '-d', t, '-q'])
            shutil.copy(glob.glob(os.path.join(t, 'reportlab-*.whl'))[0], whl)
    for f in sorted(glob.glob(os.path.join(APP, '**', '*'), recursive=True)):
        if os.path.isfile(f):
            print(f'{os.path.getsize(f) / 1e6:7.2f} MB  {os.path.relpath(f, ROOT)}')


if __name__ == '__main__':
    main()

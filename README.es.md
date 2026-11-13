# Zero Mouse Clicks

**De la imagen de una planta de mercadeo a un juego de planos arquitectónicos. Sin API en la nube y sin tokens.**

[Read in English](README.md)

Las constructoras y los portales inmobiliarios publican las plantas de los apartamentos como imágenes de colores: muebles, pisos de madera, sombras, nombres de los espacios y algunas medidas. Zero Mouse Clicks lee una de esas imágenes y dibuja el juego de planos que haría un arquitecto: muros, puertas y ventanas, ejes, cotas en cadena, áreas, cuadros, acabados de piso, una propuesta eléctrica esquemática y dos cortes.

Todo corre en tu propio computador: tres modelos de segmentación muy pequeños (ONNX, de 0,8 a 1,3 MB) más reglas explícitas. Los modelos se entrenaron **solo con plantas sintéticas** que genera el código de este repositorio, así que sobre los pesos no pesa ninguna licencia de bases de datos de terceros.

> **Estado: prototipo de investigación (v0.2).** Con algunas plantas funciona bien y con otras mal. Cada lámina lleva la nota *«Drawn from marketing plan — verify dimensions on site»* (dibujado a partir de una planta de mercadeo; verificar medidas en obra), y va en serio.

## Uso rápido

```bash
git clone https://github.com/zeromouseclicks/zero-mouse-clicks.git
cd zero-mouse-clicks
pip install -e .

# área privada (interior neta) en m², la que trae el folleto
zmc examples/synthetic_54m2.png --area 54.4 -o planos.pdf --debug

# o que la lea de la imagen (requiere Tesseract)
zmc mi_planta.jpg
```

Entre 8 y 35 segundos por planta en CPU, según el equipo y el tamaño de la imagen. No necesita GPU.

**Opcional, recomendado: [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html).** Con él se leen los nombres de los espacios (en español e inglés), las áreas declaradas y las medidas impresas de los espacios («2.69 x 2.20», «12'4" x 11'0"»), que fijan la escala. Sin él, todo funciona igual y el tipo de cada espacio se deduce por reglas.

## En el navegador

[zeromouseclicks.com](https://zeromouseclicks.com) corre este mismo pipeline en el navegador del visitante: Pyodide corre el paquete `zmc`, onnxruntime-web los tres modelos y tesseract.js lee los textos. No se sube nada y no hay servidor. La página carga `docs/app/`, que se arma desde `zmc/` y `web/` con `python web/build.py` (volver a correrlo cuando cambie el paquete).

## Qué entrega

Un PDF A3 con seis láminas: A-101 planta, A-102 cuadros de áreas y de puertas y ventanas, A-103 acabados de piso, A-104 eléctrico esquemático, A-201 corte por la cocina y A-202 corte por el baño. Las alturas y los puntos eléctricos son supuestos, no información leída de la imagen.

## Limitaciones conocidas

- Se probó con **9 plantas reales** de constructoras colombianas (no se incluyen porque son de sus dueños) y con plantas sintéticas. Las de otros países, los dibujos en blanco y negro y las fotos de planos en papel están casi sin probar.
- Algunos espacios salen con el tipo equivocado, sobre todo sin OCR.
- Las imágenes de baja resolución (menos de ~800 px) pierden muros.
- Los acabados de piso se adivinan por la textura y fallan con frecuencia.
- No hay escaleras, apartamentos de dos niveles ni muros curvos.
- Las plantas de mercadeo no vienen a escala. Si el folleto trae las medidas de los espacios, la escala sale de ellas (votación entre todas, X e Y por separado; corrige imágenes estiradas) y el plano suele quedar dentro del 1–2 %. Sin medidas impresas, la escala sale del área declarada según su definición (privada ≈ espacios netos + media partición; construida o pies² ≈ huella con muros): escribe el área privada cuando puedas.
- Los comentarios del código están en español.

## Contribuir

Los reportes y aportes son bienvenidos, sobre todo los casos en que falla. **Comparte solo plantas que sean tuyas o que tengas derecho a compartir**: los folletos de las constructoras tienen derechos de autor.

## Licencia

[Apache 2.0](LICENSE), para el código y los pesos de los modelos. Puedes usarlo, modificarlo y redistribuirlo, también con fines comerciales, conservando la licencia y el [NOTICE](NOTICE). Se entrega «tal cual», sin garantías: verifica siempre las medidas en obra. La licencia no autoriza el uso del nombre «Zero Mouse Clicks».

Creado por **Julián Páez** · Lo que habitas — IA para el sector AECO.

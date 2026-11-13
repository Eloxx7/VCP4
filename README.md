
# Práctica 4 — Detección y seguimiento de personas, vehículos y matrículas

## 1. Objetivo

El objetivo de esta práctica es desarrollar un sistema capaz de detectar y seguir personas y vehículos en un vídeo, localizar las matrículas de los vehículos y, cuando la calidad de imagen lo permite, reconocer su texto mediante OCR. Además, se comparan dos detectores de matrículas basados en aprendizaje profundo (YOLO11n y RT-DETR-L) y un método clásico de visión artificial basado en contornos.

La implementación se ha realizado en Python utilizando Ultralytics, PyTorch, OpenCV y Pandas. El entrenamiento y las pruebas se ejecutaron en un equipo con una GPU NVIDIA GeForce RTX 4060 Laptop GPU de 8 GB, Python 3.9.5, PyTorch 2.8.0+cu126 y Ultralytics 8.4.209.

## 2. Datos y preparación del conjunto de entrenamiento

Se parte de un conjunto de imágenes y anotaciones de matrículas. El cuaderno contempla la conversión automática de anotaciones desde varios formatos: Pascal VOC (`.xml`), YOLO (`.txt`), COCO (`.json`) y LabelMe (`.json`). En el procesamiento específico de los JSON utilizados, cada matrícula se transforma a una caja delimitadora y se normalizan sus coordenadas para adaptarlas al formato YOLO. Se utiliza una única clase, denominada `plate`.

La división del conjunto se realiza de forma aleatoria, fijando la semilla en 42 para hacer reproducible el reparto:

- 70 % para entrenamiento.
- 10 % para validación.
- 20 % para test.

El conjunto se organiza en directorios `train`, `val` y `test`, cada uno con sus carpetas `images` y `labels`. El archivo `plates.yaml` define las rutas y la única clase del problema.

**Acceso al conjunto de datos:** [AÑADIR ENLACE PÚBLICO AL DATASET / GOOGLE DRIVE].  
**Número de imágenes por partición:** [AÑADIR LOS VALORES MOSTRADOS AL EJECUTAR EL REPARTO].  
**Número total de matrículas anotadas:** [AÑADIR EL VALOR MOSTRADO POR EL CUADERNO].

Es importante incluir en la entrega el conjunto de datos o un enlace con permisos de acceso, además de los pesos de los modelos entrenados si se desea que la ejecución pueda reproducirse sin volver a entrenarlos.

## 3. Entrenamiento de los detectores

Se entrenan dos modelos para localizar matrículas: YOLO11n y RT-DETR-L. Ambos reciben imágenes de tamaño 640 × 640. El entrenamiento de RT-DETR-L se configura con 50 épocas y batch de 8. Para YOLO11n, el cuaderno parte del último checkpoint de una ejecución previa y realiza un entrenamiento adicional de 50 épocas con batch de 16. Por tanto, debe conservarse la distinción entre el entrenamiento inicial y el ajuste adicional de YOLO11n al documentar los resultados completos.

### 3.1. YOLO11n

YOLO11n es la alternativa más ligera de las dos. En la evaluación sobre el conjunto de test se obtiene:

- Precisión: **0,9003**.
- Recall: **0,6048**.
- mAP@0.5: **0,6228**.
- mAP@0.5:0.95: **0,3405**.
- Tiempo de inferencia: **3,9297 ms por imagen**.

El modelo tiene aproximadamente 2,60 millones de parámetros y 6,4 GFLOPs. Su principal ventaja en esta comparación es la velocidad de inferencia. Sin embargo, su recall es inferior al de RT-DETR-L, por lo que deja sin detectar una mayor proporción de las matrículas presentes.

### 3.2. RT-DETR-L

RT-DETR-L se entrena durante 50 épocas. La ejecución registrada termina correctamente y consume aproximadamente **1,447 horas**. En la última época se registran, entre otras, las pérdidas `giou_loss = 0,5092`, `cls_loss = 0,3744` y `l1_loss = 0,09179`. La validación final del mejor checkpoint obtiene:

- Precisión: **0,910**.
- Recall: **0,753**.
- mAP@0.5: **0,765**.
- mAP@0.5:0.95: **0,450**.
- Tiempo de inferencia: **12,1 ms por imagen** (medición de validación registrada por Ultralytics).

El modelo cuenta con aproximadamente 31,99 millones de parámetros y 105,3 GFLOPs. En comparación con YOLO11n, obtiene mejores métricas de detección, pero requiere más tiempo de inferencia y más recursos computacionales.

### 3.3. Evolución de las pérdidas y métricas

El cuaderno genera gráficas comparativas de la pérdida de cajas en entrenamiento, la pérdida de cajas en validación y el mAP@0.5 a lo largo de las épocas, leyendo los ficheros `results.csv` de cada ejecución.

Estas curvas permiten observar si el modelo aprende de forma progresiva y si las métricas de validación acompañan a la mejora de entrenamiento. En la ejecución de RT-DETR-L, las pérdidas y las métricas presentan pequeñas oscilaciones entre épocas, algo esperable durante el ajuste. Al finalizar, la evaluación del mejor checkpoint alcanza un mAP@0.5 de 0,765.

**Gráficas del entrenamiento:** incluir aquí las gráficas generadas en la sección 5.1 del cuaderno, o enlazar los ficheros correspondientes.
- `train/box_loss` o `train/giou_loss`.
- `val/box_loss` o `val/giou_loss`.
- `metrics/mAP50(B)`.

**Nota sobre los resultados:** las métricas finales de test sí aparecen en la salida del cuaderno. Para documentar con precisión la evolución completa de YOLO11n y RT-DETR-L, deben adjuntarse los `results.csv` originales y las gráficas generadas; no se deben reconstruir valores intermedios a partir de las métricas finales.

## 4. Comparación de YOLO11n y RT-DETR-L en test

La siguiente tabla recoge los resultados obtenidos al evaluar los mejores pesos de cada modelo sobre la partición de test:

| Modelo | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 | Inferencia (ms/imagen) |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0,9003 | 0,6048 | 0,6228 | 0,3405 | 3,9297 |
| RT-DETR-L | 0,8594 | 0,7029 | 0,6970 | 0,3955 | 22,8070 |

En esta tabla se muestran las métricas obtenidas específicamente en la evaluación de test del cuaderno. La precisión y el recall de RT-DETR-L son distintos de los valores de su validación final, ya que corresponden a particiones diferentes.

RT-DETR-L obtiene un mAP@0.5 y un mAP@0.5:0.95 superiores, además de un recall mayor. Esto indica que localiza una proporción mayor de las matrículas y presenta un rendimiento global de detección superior en este conjunto de test. YOLO11n, por su parte, consigue mayor precisión y es notablemente más rápido: su tiempo de inferencia es aproximadamente 5,8 veces menor que el de RT-DETR-L según esta medición.

Por tanto, la elección depende del objetivo: YOLO11n resulta más adecuado cuando prima la velocidad, mientras que RT-DETR-L ofrece una detección más completa cuando se puede asumir un coste de inferencia superior.

## 5. Comparación con un método clásico basado en contornos

Como alternativa a los detectores entrenados, se implementa un procedimiento clásico mediante OpenCV. El proceso consta de los siguientes pasos:

1. Conversión de la imagen a escala de grises.
2. Aplicación de un filtro bilateral para reducir ruido conservando bordes.
3. Detección de bordes mediante Canny y dilatación.
4. Extracción de contornos y aproximación poligonal.
5. Selección de candidatos con cuatro vértices, área mínima y relación de aspecto entre 2,0 y 6,5.

El método utiliza la forma aproximadamente rectangular de las matrículas como criterio principal. No aprende características a partir de ejemplos, por lo que puede verse afectado por fondos con bordes rectangulares, iluminación, perspectiva y oclusiones.

La evaluación se realiza sobre el mismo conjunto de test. Una detección se considera correcta cuando su IoU con una caja real es igual o superior a 0,5.

| Método | Precisión | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Contornos (OpenCV) | 0,124 | 0,096 | 0,108 | 101 | 716 | 949 |
| YOLO11n | 0,874 | 0,623 | 0,727 | 654 | 94 | 396 |
| RT-DETR-L | 0,781 | 0,722 | 0,750 | 758 | 213 | 292 |

Los resultados muestran una diferencia clara entre el método clásico y los modelos entrenados. El detector por contornos obtiene una precisión y un recall muy bajos, además de generar numerosos falsos positivos. Esto se debe a que la forma rectangular no es una característica suficiente para distinguir una matrícula de otros elementos de la escena.

Entre los modelos entrenados, RT-DETR-L obtiene el mayor recall y el mejor F1, mientras que YOLO11n mantiene una precisión mayor. En consecuencia, los detectores basados en aprendizaje profundo resultan mucho más adecuados para este problema.

**Gráficas y ejemplos visuales:** incluir las imágenes comparativas generadas en la sección 6 del cuaderno, donde se muestran las detecciones de contornos, YOLO11n y RT-DETR-L sobre las mismas imágenes.

No se ha registrado en el cuaderno una medición de tiempo comparable para el método de contornos; por tanto, no se incluye una cifra de velocidad para él.

## 6. Procesamiento del vídeo: seguimiento, matrículas y OCR

Una vez comparados los detectores, se construye un pipeline para procesar el vídeo de test `C0142.mp4`.

### 6.1. Detección y seguimiento

Se utiliza YOLO11n (`yolo11n.pt`) para detectar personas, coches, motocicletas, autobuses y camiones. El seguimiento entre fotogramas se realiza mediante ByteTrack, que asigna un identificador a cada objeto y permite mantener su identidad a lo largo del vídeo.

Las clases consideradas son:

| Clase | Identificador COCO |
|---|---:|
| Persona | 0 |
| Coche | 2 |
| Motocicleta | 3 |
| Autobús | 5 |
| Camión | 7 |

Para cada vehículo se recorta su región y se aplica el detector de matrículas seleccionado. En la ejecución documentada se utiliza RT-DETR-L.

### 6.2. Reconocimiento de matrículas

La lectura de caracteres se realiza mediante EasyOCR, si está disponible en el entorno. Antes del reconocimiento, el recorte de la matrícula se convierte a escala de grises, se amplía mediante interpolación cúbica y se aplica CLAHE para mejorar el contraste.

Después, se normaliza el texto reconocido mediante reglas que contemplan confusiones habituales entre caracteres, como `O` y `0`, `I` y `1` o `S` y `5`. También se comprueba que el resultado encaje con determinados patrones de matrículas.

El sistema acumula las lecturas por identificador de seguimiento y suma sus confianzas. De esta manera, el texto final de un vehículo se decide a partir de las lecturas obtenidas en distintos fotogramas, en lugar de depender de una única observación.

Durante la ejecución se produjo un error de permisos al intentar instalar EasyOCR mediante `%pip` (Windows devolvió `WinError 5` al acceder a `cv2.pyd`). No obstante, el cuaderno continuó con el bloque de OCR disponible en el entorno. Conviene comprobar que EasyOCR está correctamente instalado antes de reproducir el experimento; si no está disponible, la columna de texto de matrícula puede quedar vacía.

### 6.3. Resultados del vídeo

En la ejecución registrada se procesaron **2.832 fotogramas**. Los conteos por identificadores únicos fueron:

| Clase | Objetos detectados |
|---|---:|
| Coches | 222 |
| Autobuses | 7 |
| Personas | 38 |
| Camiones | 7 |
| Motocicletas | 4 |

Al aplicar el filtro de estabilidad de al menos 15 fotogramas, los conteos resultantes fueron:

| Clase | Objetos estables |
|---|---:|
| Coches | 138 |
| Personas | 18 |
| Motocicletas | 2 |
| Autobuses | 1 |

Además, se detectó una matrícula en **35 vehículos**. Se obtuvo texto válido para tres identificadores de seguimiento en la ejecución registrada: `341`, `626` y `678`. Estos resultados muestran que la localización de la matrícula no implica necesariamente que el OCR pueda leer correctamente sus caracteres: la resolución, el tamaño del recorte, el movimiento y la perspectiva pueden impedir una lectura fiable.

### 6.4. Vídeo de salida

El pipeline genera un vídeo anotado (`resultado.mp4`) que muestra las cajas de detección, los identificadores de seguimiento, las cajas de matrícula y el texto reconocido cuando está disponible. En la parte superior se muestra un resumen de los objetos contabilizados.

**Vídeo de test procesado:** [AÑADIR ENLACE AL VÍDEO `resultado.mp4`].  
El vídeo incluido en la entrega debe mostrar de forma visible la matrícula localizada y, cuando exista una lectura OCR válida, el texto reconocido asociado a ella.

## 7. Volcado CSV

El procesamiento genera `resultado.csv`, con una fila por objeto detectado en cada fotograma. El fichero obtenido en la ejecución registrada contiene **20.941 filas y 15 columnas**.

| Columna | Descripción |
|---|---|
| `fotograma` | Número de fotograma procesado. |
| `tipo_objeto` | Clase del objeto detectado. |
| `confianza` | Confianza de la detección del objeto. |
| `identificador_tracking` | Identificador asignado por ByteTrack. |
| `x1`, `y1`, `x2`, `y2` | Coordenadas de la caja del objeto. |
| `matricula_en_su_caso` | Indica si se ha localizado una matrícula para el objeto. |
| `confianza_matricula` | Confianza de la detección de matrícula. |
| `mx1`, `my1`, `mx2`, `my2` | Coordenadas de la caja de matrícula. |
| `texto_matricula` | Texto final reconocido para la matrícula, cuando está disponible. |

El texto final de cada vehículo se propaga a las filas de ese identificador en las que se ha detectado una matrícula. Por tanto, la columna `texto_matricula` permite consultar la matrícula asociada al objeto seguido a lo largo del vídeo.

**CSV de salida:** [AÑADIR EL ARCHIVO `resultado.csv` O SU ENLACE].  
Debe incluirse el CSV real generado por la ejecución, no únicamente una captura de su contenido.

## 8. Conclusiones

La práctica permite comparar tres aproximaciones a la detección de matrículas. El método basado en contornos presenta un rendimiento claramente insuficiente en escenas reales, ya que depende de características geométricas simples y genera muchos falsos positivos.

Los modelos entrenados ofrecen resultados considerablemente mejores. RT-DETR-L alcanza el mejor recall y F1 sobre el conjunto de test, por lo que detecta más matrículas y proporciona el mejor equilibrio entre precisión y exhaustividad según esta métrica. YOLO11n, en cambio, destaca por su velocidad y precisión, lo que lo convierte en una opción interesante cuando el tiempo de procesamiento es un factor prioritario.

En el procesamiento del vídeo, la combinación de detección de objetos con ByteTrack permite contar objetos únicos y mantener su identidad entre fotogramas. La detección de matrículas y el OCR añaden información útil al seguimiento, aunque la lectura de caracteres sigue siendo la parte más sensible a la calidad de imagen. En la ejecución registrada se localizaron matrículas en 35 vehículos, pero solo se obtuvo texto válido para tres identificadores, por lo que existe margen de mejora en el reconocimiento.

En conjunto, los resultados muestran que el aprendizaje profundo es más apropiado que el método clásico de contornos para localizar matrículas en este escenario. La elección entre YOLO11n y RT-DETR-L debe realizarse en función del equilibrio deseado entre velocidad y rendimiento de detección.

## 9. Contenido de la entrega

La entrega debe incluir, junto con los materiales de la práctica anterior:

- [ ] Este archivo `README.md` o `README.md` con la memoria y la comparativa.
- [ ] El cuaderno de ejecución (`VC_P4_local.ipynb`).
- [ ] El vídeo de test procesado (`resultado.mp4`) o un enlace accesible al mismo.
- [ ] El CSV generado (`resultado.csv`).
- [ ] El conjunto de datos utilizado o un enlace con permisos de acceso.
- [ ] Los pesos de los detectores entrenados (`best.pt`) o una forma de descargarlos.
- [ ] Las gráficas de entrenamiento y validación, incluyendo pérdidas y mAP.
- [ ] Las imágenes de ejemplo de la comparación entre contornos, YOLO11n y RT-DETR-L.

Antes de entregar, deben sustituirse los enlaces pendientes y comprobarse que el vídeo permite visualizar las matrículas detectadas y que el CSV contiene la columna `texto_matricula` con los valores reconocidos.

# Protocolo de evaluación post-eclipse

Este documento fija cómo se puntuará la predicción de `corona26` frente a una observación terrestre primaria. La predicción se publicó antes del eclipse; **este protocolo de puntuación se ha cerrado el 12 de agosto de 2026 a las 17:46:14 CEST, durante el día del eclipse, y no se presenta como preregistrado antes de la totalidad**.

## Estado y alcance

| Elemento | Estado |
|---|---|
| Predicción puntuable | Primer panel north-up de `docs/figures/corona_channels.png` en `59d3890` |
| SHA-256 del PNG de canales | `e71d6543723cb44eeea5bfa249e4aa19bbb3c69374c52135f8f483cdcc27d787` |
| Protocol lock | Cerrado hoy; registra los hashes de este documento y del manifiesto de predicción |
| Observación primaria | No seleccionada ni descargada |
| Puntuación oficial | No ejecutada |
| PSI/MAS, LASCO y rerenders | Solo exploratorios |

El objetivo no es encontrar la comparación visual más favorable, sino aplicar una transformación físicamente justificada y métricas fijas. SSIM e IoU no son métricas principales.

## Secuencia obligatoria

1. Identificar una observación terrestre candidata sin comparar su estructura coronal con la predicción ni con observaciones alternativas.
2. Bloquear la observación primaria mediante un manifiesto con origen público, hora, SHA-256, tipo de medio y metadatos instrumentales disponibles.
3. Registrar que la selección fue la primera y que no se consultaron alternativas antes del bloqueo.
4. Determinar centro, radio solar, rotación y paridad mediante WCS, efemérides, montura o astrometría independiente de la corona predicha.
5. Ejecutar `scripts/phase_g.py` para producir `outputs/preregistered_score.json`.
6. Solo después, comparar alternativas, PSI/MAS, LASCO, otros procesados o rerenders en `outputs/exploratory_analysis.json` o artefactos inequívocamente exploratorios.

Si no puede demostrarse el orden de selección, el resultado no se denominará oficial.

## Lock del artefacto predicho

La unidad puntuable es `git:59d3890526b33b8c1d4281777a3144ae78644f99:docs/figures/corona_channels.png:first-panel`. Se puntúa el **display congelado**, no una reconstrucción numérica posterior.

El recorte es explícito y de límites semiabiertos: `[left=51, top=155, right=840, bottom=944]`. Produce 789 × 789 píxeles, con centro `(394, 394)`, radio solar `140,892857 px` y semiancho de campo `2,8 R_sun`. Excluye título, subtítulo y los otros dos paneles. Estos valores están en `docs/manifests/frozen-prediction.json`; no se detectan de nuevo mediante heurísticas.

El PNG `docs/figures/corona_prediction.png` tiene SHA-256 `e6a00d1d7f24ad309fee72696c79e6939ba6b5a13e6194cb0a74db86483daa10`, pero está orientado al horizonte y no es el artefacto oficial de puntuación.

## Convención angular

Se usa el ángulo de posición astronómico (PA):

| PA | Dirección en una imagen north-up |
|---:|---|
| 0° | Norte, arriba |
| 90° | Este, izquierda |
| 180° | Sur, abajo |
| 270° | Oeste, derecha |

El PA aumenta en sentido antihorario en coordenadas cartesianas north-up. La distancia entre ángulos es la distancia circular mínima, entre 0° y 180°; por ejemplo, 359° y 1° distan 2°.

Las convenciones internas del modelo (`+Z` hacia el observador, `+Y` norte y `+X` oeste) no sustituyen esta convención de evaluación.

## Alineamiento y máscaras

Las únicas transformaciones permitidas para la observación son:

- traslación para centrar el disco lunar/solar;
- escala isotrópica a partir del radio solar medido;
- rotación fijada por efemérides, WCS, orientación de montura o astrometría;
- reflexión únicamente cuando la paridad óptica esté documentada en el manifiesto.

Quedan prohibidos:

- optimizar rotación, centro, escala, reflexión o recorte contra la correlación con la predicción;
- deformación anisótropa, warping local o registro por rasgos coronales;
- escoger después la exposición, observación o procesado que maximice el score;
- rellenar zonas sin datos o convertir métricas no disponibles en cero;
- ajustar contraste, filtrado o detección por radio o por imagen después de ver el resultado.

`rotation_deg` es la corrección, expresada en el intervalo canónico `[-180°, 180°]`, que se aplica visualmente a la imagen fuente para llevar el norte arriba. Un valor positivo gira la imagen mostrada en sentido antihorario en coordenadas de array (`x` hacia la derecha, `y` hacia abajo): `+90°` mueve un marcador situado a la derecha del centro de la fuente hasta la parte superior de la salida. No describe la orientación original de la cámara, sino la corrección que se ejecuta.

La máscara válida debe excluir saturación, obstáculos, bordes sin datos, estrellas problemáticas y artefactos instrumentales mediante criterios independientes de la predicción. Se conserva la máscara durante el remuestreo. La cobertura es la fracción de muestras válidas del anillo; un radio con cobertura inferior al 80 % queda como `not_enough_coverage` y sus métricas son `null`.

## Perfiles y streamers

Se muestrean perfiles angulares en `1,5`, `2,0` y `2,5 R_sun`, con semianchura radial fija de `0,05 R_sun`. Cada bin angular usa la mediana de sus muestras radiales válidas.

Cada perfil se normaliza por mediana y MAD sin escalar. Un MAD nulo o insuficientes muestras produce estado `degenerate`, no un perfil artificial. Después se aplica un suavizado gaussiano circular de `sigma=3°`.

Los picos se detectan circularmente con `scipy.signal.find_peaks`, prominencia fija de `0,5 MAD` y distancia mínima de `15°`. El seam 0°/360° se trata extendiendo periódicamente el perfil; no se duplica un streamer situado en el norte.

## Métricas oficiales

Por radio se publican:

- `streamer_pa_mae_deg`: media del error circular de la asignación uno-a-uno globalmente óptima;
- `precision_at_10deg`: fracción de picos predichos emparejados a un pico observado a 10° o menos;
- `recall_at_10deg`: fracción de picos observados emparejados a un pico predicho a 10° o menos;
- `angular_profile_correlation`: correlación de Pearson de los perfiles registrados, **sin optimizar rotación**;
- error medio este y oeste, y `east_minus_west_error_deg`.

La asignación óptima minimiza la suma de distancias circulares. Los picos extra o ausentes reducen precision o recall; no se fabrican errores de 0°. Si una imagen no contiene picos, el estado es `no_peaks` y las magnitudes que no estén definidas quedan en `null`.

El resumen macro es la media no ponderada de los radios disponibles. Los valores `null` no se convierten en cero. Para describir el PA MAE se usarán estos umbrales, sin alterar la métrica:

| PA MAE | Descripción |
|---:|---|
| <= 5° | Excelente |
| > 5° y <= 10° | Bueno |
| > 10° y <= 20° | Parcial |
| > 20° | Pobre |

## Contraste confirmatorio este-oeste

La predicción previa afirma que el error debe ser mayor al este que al oeste. Se publica siempre `east_minus_west_error_deg = east_error_deg - west_error_deg`, aunque sea cero, negativo o no disponible. Un valor positivo apoya la predicción; un valor no positivo la contradice. El hemisferio se asigna por el PA observado: este `[0°, 180°)` y oeste `[180°, 360°)`.

## Separación confirmatoria y exploratoria

`preregistered_score.json` solo puede contener la predicción congelada, la primera observación primaria bloqueada, los parámetros anteriores y su provenance completa. Cualquier variante posterior pertenece a `exploratory_analysis.json`, incluidas:

- PSI/MAS y LASCO;
- observaciones o exposiciones alternativas;
- rerenders, datos físicos internos o otros canales;
- métricas adicionales como SSIM o IoU;
- sensibilidad a filtros, umbrales, radios o alineamientos.

Una exploración puede explicar un fallo, pero no reemplaza ni corrige el score oficial.

## Provenance y ejecución

El protocol lock valida por SHA-256 este documento y `frozen-prediction.json`. El modo oficial acepta únicamente `docs/manifests/validation-protocol-lock.json`; su ruta y SHA-256 están anclados independientemente en código y no pueden redefinirse desde otro lock. La CLI carga el PNG directamente del objeto Git de `59d3890`; por tanto, un rerender local con el mismo nombre no puede sustituirlo.

El manifiesto de observación debe incluir como mínimo:

- rol `primary`, orden de selección `1` y marcas temporales con zona horaria;
- proveedor, instrumento y URL pública;
- ruta relativa estrictamente confinada al repositorio, tipo de medio y SHA-256 completo;
- centro, radio solar, rotación, fuente de rotación, paridad y evidencia de reflexión si procede;
- declaración explícita de que no se optimizó contra la predicción.

La ejecución futura será:

```bash
uv run python scripts/phase_g.py \
  --observation-manifest outputs/primary-observation.manifest.json
```

Hoy ese manifiesto no existe deliberadamente. La CLI debe fallar antes de escribir un score si falta, si un hash no coincide, si la selección no fue primaria o si el alineamiento admite hindsight.

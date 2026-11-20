# Predicción del precio del petróleo Brent con RNN, LSTM y GRU

Caso II de Aprendizaje Automático (Ingeniería Matemática, UAX). Trabajo individual.

Se predice el precio del Brent a **5, 63 y 252 días hábiles** (≈ 1 semana, 3 meses y 1 año), con un
**rango de incertidumbre del 80 %**, y se compara una RNN simple, una LSTM y una GRU en igualdad de
condiciones frente al modelo ingenuo (el precio futuro es igual al de hoy).

**Web de la herramienta:** https://alvaroglezfdez.github.io/oil_price_prediction/

---

## Resultados

Test final: muestras con fecha t entre el 03-01-2023 y el 10-11-2026 (usado una sola vez). Errores en
dólares sobre el precio previsto P̂ₜ₊ₕ = Pₜ·exp(ŷ). Las redes usan la **previsión media de 3 semillas**
(la que usa la web); entre paréntesis, la media de las métricas de cada semilla por separado.

| h | Modelo | MAE ($) | RMSE ($) | Mejora vs ingenuo | DM (p-valor) | Sesgo ($) | Acierto dirección | Cobertura 80 % | Parámetros |
|---|---|---|---|---|---|---|---|---|---|
| 5 | **RNN** | 3,71 | 5,48 (5,51) | +0,8 % (+0,3 %) | −1,03 (0,30) | −0,06 | 51,8 % | 77,2 % | 1.635 |
| 5 | LSTM | 3,73 | 5,48 (5,51) | +0,8 % (+0,2 %) | −0,31 (0,76) | −0,02 | 52,2 % | 79,2 % | 6.243 |
| 5 | GRU | 3,82 | 5,63 (5,66) | −2,0 % (−2,4 %) | 1,49 (0,14) | 0,70 | 51,1 % | 75,6 % | 4.707 |
| 5 | Ingenuo | 3,71 | 5,52 | — | — | −0,08 | (sube 50,4 %) | 78,1 % | 0 |
| 63 | **RNN** | 11,81 | 16,46 (17,71) | +7,5 % (+0,5 %) | −0,91 (0,36) | −0,68 | 61,3 % | 85,7 % | 1.635 |
| 63 | LSTM | 14,85 | 19,95 (20,35) | −12,1 % (−14,3 %) | 1,02 (0,31) | −0,67 | 42,6 % | 84,1 % | 6.243 |
| 63 | GRU | 14,14 | 18,14 (18,82) | −1,9 % (−5,7 %) | 0,20 (0,84) | 6,38 | 51,0 % | 68,3 % | 4.707 |
| 63 | Ingenuo | 11,58 | 17,80 | — | — | −2,09 | (sube 46,2 %) | 87,6 % | 0 |
| 252 | **RNN** | 15,70 | 19,57 (25,66) | +9,9 % (−18,1 %) | −0,50 (0,62) | 5,12 | 60,2 % | 79,4 % | 1.635 |
| 252 | LSTM | 29,38 | 32,67 (35,17) | −50,4 % (−61,9 %) | 1,59 (0,11) | 22,73 | 47,5 % | 99,5 % | 6.243 |
| 252 | GRU | 27,80 | 30,76 (32,20) | −41,6 % (−48,3 %) | 1,18 (0,24) | 20,39 | 39,6 % | 82,3 % | 4.707 |
| 252 | Ingenuo | 15,72 | 21,72 | — | — | −3,47 | (sube 40,9 %) | 97,4 % | 0 |

- **Mejora vs ingenuo** = 1 − RMSE_red / RMSE_ingenuo. **DM**: contraste de Diebold-Mariano frente al
  ingenuo con pérdida cuadrática en dólares y varianza HAC de Newey-West con h−1 retardos (los errores de
  horizontes solapados están autocorrelacionados); DM < 0 favorece a la red. **Sesgo** = media de previsto − real.
- Tiempo (s), ajuste de hiperparámetros + entrenamiento final: RNN 565 + 81, LSTM 301 + 36, GRU 862 + 102
  (CPU). Tabla completa en [results/comparativa.csv](results/comparativa.csv).

**Conclusión.** Ninguna red supera al ingenuo de forma significativa en ningún horizonte (todos los p-valores
de DM > 0,10). La incertidumbre crece mucho con el horizonte (el rango del 80 % pasa de ±6 % a 5 días a
−23 % / +64 % a un año). El valor de la herramienta no está en la cifra central, sino en **los rangos de
riesgo** y en saber **cuándo no fiarse**. Es el resultado esperable en un mercado con precios cercanos a un
paseo aleatorio, y se presenta tal cual.

### Por qué la RNN (y no la GRU elegida por validación)

La selección por validación (pérdida media en 2018-2021) favorecía la **GRU**, pero esa media la domina el
**pliegue 2020** (COVID), con pérdidas unas 10 veces mayores que el resto (4,5-5,2 frente a 0,4-1,7): en la
práctica se elegía la red que mejor aguantaba el COVID. En test, la GRU fue la peor a 5 días. Para la
herramienta se usa la **RNN simple (32 unidades, dropout 0,2, 12 épocas)** porque:

1. es la única red que iguala o mejora ligeramente al ingenuo a 5 y 63 días;
2. es la más estable entre semillas a 5 y 63 días;
3. es la de menor coste (1.635 parámetros, frente a 4.707 de la GRU y 6.243 de la LSTM).

Esta decisión se tomó viendo el test, así que **el test deja de ser una estimación sin sesgo para el modelo
elegido**. Se documenta con transparencia; ninguna red es significativamente distinta del ingenuo, así que
la elección no cambia la conclusión.

---

## Arquitectura

```
Python (motor)                     GitHub Actions                      Web estática (GitHub Pages)
src/download.py    ─┐              actualizar_datos.yml (L-V 06:30)    web/index.html + app.js
src/produccion.py   ├─► JSON ──►   reentrenar.yml (día 1, 05:00)  ──►  lee web/data/*.json
src/exportar_web.py ┘  web/data/   pages.yml (publica web/)            (no calcula nada)
```

- **Motor en Python**: descarga, variables, modelo y predicción. Exporta a JSON en `web/data/`.
- **Web estática** (HTML/CSS/JS, Plotly.js 2.35.2 desde CDN): solo lee los JSON. No hay servidor.
- **Automatización**: cada día laborable se descargan los datos y se recalcula la previsión con los pesos
  guardados; el día 1 de cada mes se reentrena el modelo de producción con todos los datos (misma
  configuración, sin volver a buscar hiperparámetros). `pages.yml` publica la web tras cada actualización.
- **Por qué el reentrenamiento se lanza desde Actions y no desde la web**: una web estática no puede
  ejecutar Python ni PyTorch, y permitir que cualquier visitante lanzara un reentrenamiento exigiría un
  servidor y credenciales con permiso de escritura en el repo. Desde Actions el proceso es reproducible,
  queda registrado y solo lo puede lanzar el propietario (botón *Run workflow* en la pestaña Actions).

---

## Decisiones de diseño

- **Objetivo en log-rentabilidad**: yₕ(t) = ln(Pₜ₊ₕ / Pₜ). Es aproximadamente estacionario (ADF/KPSS, ver
  [reports/eda/](reports/eda/)), es comparable entre épocas de precios muy distintos y el precio previsto
  siempre es positivo: P̂ = Pₜ·exp(ŷ).
- **Una red por arquitectura con 3 salidas** (5, 63, 252). Pérdida MSE sobre los objetivos estandarizados por
  horizonte (con la media y desviación de train) para que el horizonte de 252 días no domine. Pérdida con
  máscara: las muestras recientes sin etiqueta a 252 días se aprovechan para los horizontes que sí conocen.
- **Entradas**: ventana de 60 días hábiles de 14 variables aproximadamente estacionarias: log-rentabilidades
  de Brent, WTI, futuro WTI, gas (NG=F), cobre, oro, índice dólar, USD/EUR y CNY/USD; ln(VIX); tipo BCE (nivel
  y cambio); volatilidad de 20 días del Brent; distancia ln(Pₜ) − ln(media móvil de 252 días).
- **Arquitecturas idénticas salvo la celda**: `nn.RNN(tanh)` / `nn.LSTM` / `nn.GRU`, 1 capa → dropout →
  `Linear(units, 3)`. Adam (lr 1e-3), batch 64, máximo 100 épocas, early stopping (paciencia 10, restaurando
  los mejores pesos), gradient clipping 1,0, semillas fijas, CPU.
- **Mismo presupuesto de hiperparámetros** para las tres: units ∈ {32, 64} × dropout ∈ {0, 0,2}.
- **Walk-forward expansivo con purga de 252 días hábiles**: validación en 2018, 2019, 2020 y 2021; el
  entrenamiento termina 252 días hábiles antes del inicio de la validación, porque la etiqueta mira hasta 252
  días al futuro. Particiones exactas en [results/particiones.csv](results/particiones.csv).
- **Winsorización (0,5/99,5) y escalado ajustados solo con el train de cada partición**. El objetivo no se
  winsoriza. El tipo BCE (nivel y cambio) se excluye de la winsorización: su cambio diario es distinto de 0
  solo el 0,66 % de los días y los percentiles valen 0, lo que anularía la variable.
- **Test usado una sola vez** (t ≥ 2023-01-01, con purga respecto al entrenamiento final, que termina el
  31-12-2021). El modelo final se entrena durante la mediana de las épocas óptimas de los pliegues, con 3 semillas.
- **Intervalos del 80 % a partir de residuos out-of-fold**: cuantiles 10 y 90 de (y real − y predicha) en los
  años de validación, para cada red y horizonte; intervalo = ŷ + [q₁₀, q₉₀]. Para el ingenuo, los cuantiles
  de y real en esos mismos años.

---

## Notas sobre los datos

- **Fuentes**: FRED (Brent DCOILBRENTEU, WTI, tipo BCE, USD/EUR, CNY/USD, VIX) y Yahoo Finance (futuros
  BZ=F y CL=F, gas NG=F, cobre, oro, índice dólar). Datos diarios desde 2000-01-01.
- **Corrección del calendario del script original** ([scripts/descarga_datos_petroleo.py](scripts/descarga_datos_petroleo.py)):
  hacía `ffill()` sobre la unión de todas las fechas y después filtraba por Brent no nulo, filtro que ya no
  eliminaba nada. Eso creaba **500 días falsos del Brent** (festivos de Londres con el precio anterior
  repetido). En [src/download.py](src/download.py) el calendario son los días con dato real del Brent y el
  resto de series se rellenan solo hacia delante.
- **Gas: NG=F (Yahoo) en lugar de FRED DHHNGSP**, que entre 2000 y 2006 solo tiene ~60 observaciones al año.
- **BZ=F (futuro del Brent) excluido de las entradas**: solo existe desde julio de 2007; se mantiene en el dataset.
- **WTI negativo** (20-04-2020, −36,98 $): los valores ≤ 0 se marcan como faltantes antes de calcular rentabilidades.
- **Cesta OPEP excluida**: no tiene una fuente diaria gratuita y fiable.
- Outliers reales que no se eliminan: Brent −64 % el 21-04-2020, máximo de 143,9 $ el 03-07-2008, picos de
  2022 y de 2026 (hasta 138,2 $).

---

## Limitaciones

- **Sobreajuste muy temprano**: la época óptima cae casi siempre entre la 1 y la 5 ([reports/training/](reports/training/)).
  La pérdida de validación sube en cuanto la de train empieza a bajar: hay poca señal más allá del paseo aleatorio.
- **El pliegue 2020 domina la selección de hiperparámetros**: su pérdida es unas 10 veces mayor.
- **Pocas observaciones independientes a 252 días**: unos 3 años de objetivos solapados en test, y alta
  variabilidad entre semillas (RMSE de cada semilla ± 3-5 $). La previsión media de 3 semillas es mejor que
  cada semilla por separado.
- **Coberturas irregulares a 63 y 252 días**: desde el 68 % (GRU, 63 días) al 99,5 % (LSTM, 252 días). Los
  intervalos incluyen el choque de 2020 y no se adaptan a la volatilidad del momento.
- **Selección del modelo de la web viendo el test** (ver arriba).
- **Tiempo de la LSTM corregido**: la ejecución LSTM u32/dropout 0/pliegue 2020 registró 10.275,7 s porque el
  PC se suspendió (el log salta de 19:57 a 22:48). Repetida con la misma semilla dio exactamente el mismo
  resultado (pérdida 4,9331, época óptima 11) en 15,6 s; se corrigió con
  [scripts/corregir_tiempo_lstm.py](scripts/corregir_tiempo_lstm.py) en `tuning.csv`, `lstm_meta.json` y
  `comparativa.csv`. El log original se conserva sin modificar.

## Trabajo futuro

- Redes más pequeñas y más regularización (weight decay, dropout recurrente).
- Pérdida robusta (Huber) para reducir el peso de los choques extremos.
- Combinación con el ingenuo (contracción de la previsión hacia 0).
- Modelos por horizonte, o modelos de reversión a la media para el largo plazo.
- Intervalos adaptados a la volatilidad (GARCH/EWMA) en lugar de cuantiles fijos.

---

## Estructura del repositorio

```
├── config.yaml              # todos los parámetros (datos, variables, partición, entrenamiento, producción)
├── src/
│   ├── download.py          # descarga FRED + Yahoo y alineación al calendario del Brent
│   ├── eda.py               # análisis exploratorio -> reports/eda/
│   ├── features.py          # variables de entrada y objetivo
│   ├── windows.py           # ventanas, partición walk-forward con purga y preprocesado
│   ├── models.py            # RNN/LSTM/GRU y bucle de entrenamiento
│   ├── train.py             # ajuste de hiperparámetros + entrenamiento final
│   ├── evaluate.py          # intervalos, métricas de test, Diebold-Mariano, gráficos
│   ├── produccion.py        # modelo de producción y predicción actual
│   └── exportar_web.py      # JSON para la web
├── scripts/                 # scripts del profesor (referencia) y corrección del tiempo de la LSTM
├── data/raw, data/processed # descargas sin tocar y dataset alineado
├── models/                  # modelos de la comparativa y models/produccion/
├── results/                 # CSV/JSON de resultados y logs
├── reports/                 # gráficos del EDA, curvas de aprendizaje y predicciones de test
├── web/                     # web estática (index.html, styles.css, app.js, data/)
├── docs/assignment/         # enunciado
└── .github/workflows/       # actualizar_datos.yml, reentrenar.yml, pages.yml
```

## Cómo ejecutarlo

**Con Poetry** (`pyproject.toml` + `poetry.lock`; Python 3.11-3.14):

```bash
poetry install
poetry run python -m src.download        # datos
poetry run python -m src.eda             # análisis exploratorio
poetry run python -m src.train           # ajuste + entrenamiento final (~1 h en CPU)
poetry run python -m src.evaluate        # métricas de test, DM, gráficos
poetry run python -m src.produccion      # modelo de producción + predicción (--solo-prediccion para no reentrenar)
poetry run python -m src.exportar_web    # JSON de la web
```

> Con Poetry, torch se instala desde PyPI: en Linux incluye CUDA y la descarga es grande. La alternativa con
> pip usa la versión solo CPU.

**Con pip** (vía que usan GitHub Actions):

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.download            # y el resto de pasos igual que arriba
```

`python -m src.train --rapido` hace una prueba de humo (1 pliegue, 2 épocas) en carpetas `prueba/`.

**Ver la web en local** (hay que servirla; abrir el HTML directamente no deja leer los JSON):

```bash
cd web
python -m http.server 8000        # y abrir http://localhost:8000
```

## Automatización (GitHub Actions)

| Workflow | Cuándo | Qué hace |
|---|---|---|
| `actualizar_datos.yml` | L-V 06:30 UTC + manual | descarga, `src.produccion --solo-prediccion`, `src.exportar_web`, commit y push |
| `reentrenar.yml` | día 1 de cada mes 05:00 UTC + manual | descarga, reentrena el modelo de producción, exporta, commit y push |
| `pages.yml` | push que cambie `web/`, tras los dos anteriores, o manual | publica `web/` en GitHub Pages |

Para lanzarlos a mano: pestaña **Actions** → workflow → **Run workflow**.

## Agradecimientos

La idea de incluir el contraste de Diebold-Mariano se comentó con una compañera, con su permiso; la
implementación es propia.

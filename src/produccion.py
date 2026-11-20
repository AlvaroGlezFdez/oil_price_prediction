"""
Modelo de producción y predicción actual (la que muestra la web).

* Entrenamiento (por defecto): entrena la red elegida (config.yaml -> produccion) con TODOS los
  datos disponibles. El preprocesado se ajusta con ese mismo entrenamiento. Las muestras más
  recientes no tienen todas las etiquetas (p. ej. falta y_252 en el último año): la pérdida con
  máscara las aprovecha para los horizontes que sí conocen. No se buscan hiperparámetros.
  -> models/produccion/<red>_s<semilla>.pt, preprocesado.json, meta.json
* Predicción: para el último día disponible y h = 5, 63, 252: precio actual, precio previsto
  (media de semillas), variación %, intervalo del 80 % (cuantiles de los residuos OOF de
  results/intervalos.json) y la referencia del ingenuo con su intervalo.
  -> results/prediccion_actual.json

Uso:
  python -m src.produccion                    reentrena y predice
  python -m src.produccion --solo-prediccion  usa los pesos guardados y solo predice
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from src.config import RAIZ, cargar_config
from src.features import construir_entradas, construir_objetivo
from src.models import RedRecurrente, entrenar, fijar_semillas_torch, predecir
from src.windows import Preprocesador, construir_ventanas, posiciones_validas

ETIQUETAS = {5: "corto plazo (5 días hábiles, ~1 semana)",
             63: "medio plazo (63 días hábiles, ~3 meses)",
             252: "largo plazo (252 días hábiles, ~1 año)"}


def cargar_datos(cfg: dict):
    datos = pd.read_csv(RAIZ / cfg["rutas"]["processed"] / "dataset.csv", index_col=0, parse_dates=True)
    X = construir_entradas(datos, cfg)
    Y = construir_objetivo(datos["brent"], cfg["preprocesado"]["horizontes"])
    return datos, X, Y, posiciones_validas(X, cfg["preprocesado"]["ventana"])


def entrenar_produccion(cfg: dict, datos, X, Y, pos, dir_modelos) -> None:
    cp, cprod = cfg["preprocesado"], cfg["produccion"]
    v = cp["ventana"]
    # Todas las muestras con al menos una etiqueta conocida
    pos_train = pos[Y.iloc[pos].notna().any(axis=1).to_numpy()]
    prep = Preprocesador(cp["winsor_inf"], cp["winsor_sup"], cp["winsor_excluir"])
    prep.ajustar(X, Y, pos_train, v)
    X_tr = construir_ventanas(prep.transformar_X(X), pos_train, v)
    Y_tr = prep.estandarizar_y(Y.to_numpy()[pos_train])

    segundos = []
    for s in cprod["semillas"]:
        fijar_semillas_torch(s)
        modelo = RedRecurrente(cprod["red"], X.shape[1], cprod["units"], cprod["dropout"])
        res = entrenar(modelo, X_tr, Y_tr, cfg["entrenamiento"], s, epocas=cprod["epocas"])
        torch.save(modelo.state_dict(), dir_modelos / f"{cprod['red']}_s{s}.pt")
        segundos.append(res["segundos"])
        print(f"  semilla {s}: pérdida train final {res['historial']['train'][-1]:.4f} ({res['segundos']:.1f} s)")

    (dir_modelos / "preprocesado.json").write_text(json.dumps(prep.a_dict(), indent=2), encoding="utf-8")
    meta = {
        "red": cprod["red"], "units": cprod["units"], "dropout": cprod["dropout"],
        "epocas": cprod["epocas"], "semillas": cprod["semillas"],
        "columnas_entrada": list(X.columns), "horizontes": cp["horizontes"], "ventana": v,
        "train_inicio": str(datos.index[pos_train[0]].date()),
        "train_fin": str(datos.index[pos_train[-1]].date()),
        "n_train": int(len(pos_train)),
        "n_etiquetas_por_horizonte": {c: int(Y[c].iloc[pos_train].notna().sum()) for c in Y.columns},
        "fecha_datos": str(datos.index[-1].date()),
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segundos_entrenamiento": float(np.sum(segundos)),
    }
    (dir_modelos / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def predecir_actual(cfg: dict, datos, X, pos, dir_modelos) -> dict:
    meta = json.loads((dir_modelos / "meta.json").read_text(encoding="utf-8"))
    prep = Preprocesador.desde_dict(json.loads((dir_modelos / "preprocesado.json").read_text(encoding="utf-8")))
    intervalos = json.loads((RAIZ / cfg["rutas"]["results"] / "intervalos.json").read_text(encoding="utf-8"))
    v, horizontes, red = meta["ventana"], meta["horizontes"], meta["red"]

    t = pos[-1:]                                   # último día con ventana completa
    fecha_t = datos.index[t[0]]
    precio_t = float(datos["brent"].iloc[t[0]])
    X_t = construir_ventanas(prep.transformar_X(X), t, v)

    por_semilla = []
    for s in meta["semillas"]:
        modelo = RedRecurrente(red, X.shape[1], meta["units"], meta["dropout"])
        modelo.load_state_dict(torch.load(dir_modelos / f"{red}_s{s}.pt", weights_only=True))
        por_semilla.append(prep.desestandarizar_y(predecir(modelo, X_t))[0])
    por_semilla = np.array(por_semilla)            # (n_semillas, 3)
    y_hat = por_semilla.mean(axis=0)

    salida = {
        "fecha_datos": str(fecha_t.date()), "precio_actual": precio_t,
        "modelo": f"{red.upper()} ({meta['units']} unidades, dropout {meta['dropout']}, "
                  f"{meta['epocas']} épocas, media de {len(meta['semillas'])} semillas)",
        "fecha_entrenamiento": meta["fecha_entrenamiento"],
        "fecha_generacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nivel_intervalo": 0.80, "horizontes": [],
    }
    for j, h in enumerate(horizontes):
        q, qi = intervalos[red][str(h)], intervalos["ingenuo"][str(h)]
        # Fecha objetivo aproximada: h días hábiles (lunes a viernes) después de t
        fecha_obj = (fecha_t + pd.offsets.BDay(h)).date()
        salida["horizontes"].append({
            "h": h, "etiqueta": ETIQUETAS.get(h, f"{h} días hábiles"),
            "fecha_objetivo_aprox": str(fecha_obj),
            "log_rentabilidad_prevista": float(y_hat[j]),
            "precio_previsto": precio_t * float(np.exp(y_hat[j])),
            "variacion_pct": 100 * float(np.expm1(y_hat[j])),
            "intervalo_80": [precio_t * float(np.exp(y_hat[j] + q["q_inf"])),
                             precio_t * float(np.exp(y_hat[j] + q["q_sup"]))],
            "precio_previsto_por_semilla": [precio_t * float(np.exp(x)) for x in por_semilla[:, j]],
            "ingenuo": {"precio_previsto": precio_t,
                        "intervalo_80": [precio_t * float(np.exp(qi["q_inf"])),
                                         precio_t * float(np.exp(qi["q_sup"]))]},
        })
    return salida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-prediccion", action="store_true", help="no reentrena; usa los pesos guardados")
    args = ap.parse_args()

    cfg = cargar_config()
    dir_modelos = RAIZ / cfg["rutas"]["models"] / "produccion"
    dir_modelos.mkdir(parents=True, exist_ok=True)
    datos, X, Y, pos = cargar_datos(cfg)

    if not args.solo_prediccion:
        print(f"Entrenando modelo de producción con datos hasta {datos.index[-1].date()}...")
        entrenar_produccion(cfg, datos, X, Y, pos, dir_modelos)

    pred = predecir_actual(cfg, datos, X, pos, dir_modelos)
    ruta = RAIZ / cfg["rutas"]["results"] / "prediccion_actual.json"
    ruta.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBrent el {pred['fecha_datos']}: {pred['precio_actual']:.2f} $")
    for d in pred["horizontes"]:
        li, ls = d["intervalo_80"]
        print(f"  h={d['h']:>3} ({d['fecha_objetivo_aprox']}): {d['precio_previsto']:.2f} $ "
              f"({d['variacion_pct']:+.1f} %), intervalo 80 % [{li:.2f}, {ls:.2f}] | "
              f"ingenuo {d['ingenuo']['precio_previsto']:.2f} [{d['ingenuo']['intervalo_80'][0]:.2f}, "
              f"{d['ingenuo']['intervalo_80'][1]:.2f}]")
    print(f"Guardado en {ruta}")


if __name__ == "__main__":
    main()

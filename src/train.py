"""
Ajuste de hiperparámetros (walk-forward) y entrenamiento final (Fase 2).

1. Construye variables, objetivo, muestras válidas y particiones (results/particiones.csv).
2. Ajuste: para cada red (RNN, LSTM, GRU) y cada configuración de la rejilla
   (units x dropout), entrena en los 4 pliegues con early stopping. Métrica de selección:
   pérdida media de validación (estandarizada, media de los 3 horizontes) entre pliegues.
   -> results/tuning.csv, results/oof_predictions.csv, reports/training/curvas_<red>.png
3. Entrenamiento final: con la mejor configuración de cada red, reentrena con todo el
   entrenamiento final durante la mediana de las épocas óptimas, con 3 semillas.
   -> models/<red>_s<semilla>.pt, models/<red>_meta.json, models/preprocesado.json

Uso:
  python -m src.train            ejecución completa
  python -m src.train --rapido   prueba de humo (1 pliegue, 2 épocas, 1 config, 1 semilla)
                                 que escribe en results/prueba, models/prueba, reports/prueba
"""
import argparse
import json
import logging
import time
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import RAIZ, cargar_config
from src.features import construir_entradas, construir_objetivo
from src.models import RedRecurrente, contar_parametros, entrenar, fijar_semillas_torch, predecir
from src.windows import (Preprocesador, construir_ventanas, crear_particiones,
                         posiciones_validas, tabla_particiones)

AZUL, NARANJA, TINTA_2, REJILLA, SUPERFICIE = "#2a78d6", "#eb6834", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------
# Utilidades compartidas con evaluate.py
# ---------------------------------------------------------------
def directorios(cfg: dict, rapido: bool) -> dict:
    """Carpetas de salida (subcarpeta 'prueba' en modo rápido)."""
    sub = "prueba" if rapido else ""
    d = {
        "results": RAIZ / cfg["rutas"]["results"] / sub,
        "models": RAIZ / cfg["rutas"]["models"] / sub,
        "training": RAIZ / "reports" / (sub or "training"),
        "figuras": RAIZ / "reports" / (sub or "results"),
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def preparar_datos(cfg: dict):
    """Carga el dataset y devuelve (datos, X, Y, posiciones válidas, particiones)."""
    datos = pd.read_csv(RAIZ / cfg["rutas"]["processed"] / "dataset.csv", index_col=0, parse_dates=True)
    X = construir_entradas(datos, cfg)
    Y = construir_objetivo(datos["brent"], cfg["preprocesado"]["horizontes"])
    pos = posiciones_validas(X, cfg["preprocesado"]["ventana"])
    particiones = crear_particiones(datos.index, pos, Y, cfg)
    return datos, X, Y, pos, particiones


def preparar_particion(X: pd.DataFrame, Y: pd.DataFrame, part, cfg: dict):
    """Ajusta el preprocesado con el train de la partición y construye los tensores."""
    cp = cfg["preprocesado"]
    v = cp["ventana"]
    prep = Preprocesador(cp["winsor_inf"], cp["winsor_sup"], cp["winsor_excluir"])
    prep.ajustar(X, Y, part.pos_train, v)
    Xn = prep.transformar_X(X)
    Yv = Y.to_numpy()
    return {
        "prep": prep,
        "X_tr": construir_ventanas(Xn, part.pos_train, v),
        "Y_tr": prep.estandarizar_y(Yv[part.pos_train]),
        "X_ev": construir_ventanas(Xn, part.pos_eval, v),
        "Y_ev": prep.estandarizar_y(Yv[part.pos_eval]),
    }


def configurar_log(ruta) -> logging.Logger:
    log = logging.getLogger("entrenamiento")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(ruta, mode="w", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def nombre_config(units: int, dropout: float) -> str:
    return f"u{units}_d{dropout}"


# ---------------------------------------------------------------
# Curvas de aprendizaje
# ---------------------------------------------------------------
def grafico_curvas(red: str, config: str, historiales: dict, destino) -> None:
    """Pérdida de train y validación por época en cada pliegue (mejor configuración)."""
    n = len(historiales)
    cols = min(n, 2)
    filas = int(np.ceil(n / cols))
    fig, ejes = plt.subplots(filas, cols, figsize=(6 * cols, 3.6 * filas), squeeze=False,
                             facecolor=SUPERFICIE)
    for ax, (pliegue, h) in zip(ejes.flat, historiales.items()):
        ep = np.arange(1, len(h["train"]) + 1)
        ax.set_facecolor(SUPERFICIE)
        ax.plot(ep, h["train"], color=AZUL, lw=2, label="train (con dropout)")
        ax.plot(ep, h["val"], color=NARANJA, lw=2, label="validación")
        mejor = int(np.argmin(h["val"])) + 1
        ax.axvline(mejor, color=TINTA_2, lw=1, ls="--")
        ax.text(mejor, ax.get_ylim()[1], f" época óptima {mejor}", va="top", fontsize=8, color=TINTA_2)
        ax.set_title(pliegue.replace("pliegue_", "Validación "), loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("época")
        ax.set_ylabel("MSE estandarizado")
        ax.grid(color=REJILLA, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    for ax in list(ejes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Curvas de aprendizaje — {red.upper()} ({config})", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="prueba de humo")
    args = ap.parse_args()

    cfg = cargar_config()
    ce = dict(cfg["entrenamiento"])
    semilla = cfg["semilla"]
    horizontes = cfg["preprocesado"]["horizontes"]
    dirs = directorios(cfg, args.rapido)
    log = configurar_log(dirs["results"] / "entrenamiento.log")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    rejilla = list(product(ce["rejilla"]["units"], ce["rejilla"]["dropout"]))
    semillas_finales = ce["semillas_finales"]
    if args.rapido:
        ce["max_epocas"] = 2
        rejilla = rejilla[-1:]          # la configuración más grande (peor caso de tiempo)
        semillas_finales = semillas_finales[:1]

    # 1) Datos y particiones ------------------------------------
    datos, X, Y, pos, particiones = preparar_datos(cfg)
    cal = datos.index
    pliegues = particiones[:-1][:1] if args.rapido else particiones[:-1]
    final = particiones[-1]
    tabla = tabla_particiones(particiones, cal, Y)
    tabla.to_csv(dirs["results"] / "particiones.csv", index=False)
    log.info("Variables de entrada (%d): %s", X.shape[1], ", ".join(X.columns))
    log.info("Primera muestra válida: %s | última: %s", cal[pos[0]].date(), cal[pos[-1]].date())
    log.info("Particiones:\n%s", tabla.to_string(index=False))
    log.info("Hilos de torch: %d | rejilla: %s | max_epocas: %d",
             torch.get_num_threads(), rejilla, ce["max_epocas"])

    tensores = {p.nombre: preparar_particion(X, Y, p, cfg) for p in pliegues}
    n_entradas = X.shape[1]

    # 2) Ajuste de hiperparámetros -------------------------------
    filas, historiales, oof = [], {}, {}
    t_ajuste = time.perf_counter()
    for red in ce["redes"]:
        for units, dropout in rejilla:
            conf = nombre_config(units, dropout)
            for p in pliegues:
                T = tensores[p.nombre]
                fijar_semillas_torch(semilla)
                modelo = RedRecurrente(red, n_entradas, units, dropout)
                res = entrenar(modelo, T["X_tr"], T["Y_tr"], ce, semilla, T["X_ev"], T["Y_ev"])
                n_par = contar_parametros(modelo)
                filas.append({
                    "red": red, "config": conf, "units": units, "dropout": dropout,
                    "pliegue": p.nombre, "perdida_val": res["perdida_val"],
                    "epoca_optima": res["epoca_optima"], "epocas_ejecutadas": res["epocas_ejecutadas"],
                    "segundos": res["segundos"], "n_parametros": n_par,
                    "n_train": len(p.pos_train), "n_val": len(p.pos_eval),
                })
                historiales[(red, conf, p.nombre)] = res["historial"]
                # Predicciones de validación (out-of-fold) en log-rentabilidad
                y_hat = T["prep"].desestandarizar_y(predecir(modelo, T["X_ev"]))
                oof[(red, conf, p.nombre)] = y_hat
                log.info("%-4s %-9s %-13s val=%.4f  época óptima=%3d/%3d  %.1fs  (%d parám.)",
                         red, conf, p.nombre, res["perdida_val"], res["epoca_optima"],
                         res["epocas_ejecutadas"], res["segundos"], n_par)
    log.info("Ajuste terminado en %.1f min", (time.perf_counter() - t_ajuste) / 60)

    tuning = pd.DataFrame(filas)
    tuning.to_csv(dirs["results"] / "tuning.csv", index=False)
    resumen = (tuning.groupby(["red", "config"])
               .agg(perdida_val_media=("perdida_val", "mean"), perdida_val_desv=("perdida_val", "std"),
                    epoca_mediana=("epoca_optima", "median"), segundos=("segundos", "sum"),
                    n_parametros=("n_parametros", "first"))
               .reset_index())
    resumen.to_csv(dirs["results"] / "tuning_resumen.csv", index=False)
    log.info("Resumen del ajuste:\n%s", resumen.to_string(index=False))
    mejores = resumen.loc[resumen.groupby("red")["perdida_val_media"].idxmin()].set_index("red")

    # OOF de la mejor configuración y curvas de aprendizaje
    filas_oof = []
    for red, m in mejores.iterrows():
        hist_red = {}
        for p in pliegues:
            y_hat = oof[(red, m["config"], p.nombre)]
            hist_red[p.nombre] = historiales[(red, m["config"], p.nombre)]
            fechas = cal[p.pos_eval]
            for j, h in enumerate(horizontes):
                filas_oof.append(pd.DataFrame({
                    "red": red, "config": m["config"], "pliegue": p.nombre, "fecha": fechas,
                    "horizonte": h, "precio_t": datos["brent"].iloc[p.pos_eval].to_numpy(),
                    "y_real": Y.iloc[p.pos_eval, j].to_numpy(), "y_pred": y_hat[:, j],
                }))
        grafico_curvas(red, m["config"], hist_red, dirs["training"] / f"curvas_{red}.png")
    pd.concat(filas_oof).to_csv(dirs["results"] / "oof_predictions.csv", index=False)

    # 3) Entrenamiento final ------------------------------------
    T = preparar_particion(X, Y, final, cfg)
    with open(dirs["models"] / "preprocesado.json", "w", encoding="utf-8") as f:
        json.dump(T["prep"].a_dict(), f, indent=2)

    filas_final = []
    for red, m in mejores.iterrows():
        sel =tuning[(tuning.red == red) & (tuning.config == m["config"])]
        units, dropout = int(sel.units.iloc[0]), float(sel.dropout.iloc[0])
        epocas = max(1, int(np.round(sel.epoca_optima.median())))
        segundos = []
        for s in semillas_finales:
            fijar_semillas_torch(s)
            modelo = RedRecurrente(red, n_entradas, units, dropout)
            res = entrenar(modelo, T["X_tr"], T["Y_tr"], ce, s, epocas=epocas)
            torch.save(modelo.state_dict(), dirs["models"] / f"{red}_s{s}.pt")
            segundos.append(res["segundos"])
            filas_final.append({"red": red, "semilla": s, "epocas": epocas,
                                "perdida_train_final": res["historial"]["train"][-1],
                                "segundos": res["segundos"]})
            log.info("FINAL %-4s %s semilla=%d épocas=%d  train=%.4f  %.1fs", red, m["config"], s,
                     epocas, res["historial"]["train"][-1], res["segundos"])
        meta = {
            "red": red, "config": m["config"], "units": units, "dropout": dropout,
            "epocas_finales": epocas, "epocas_optimas_pliegues": sel.epoca_optima.tolist(),
            "semillas": semillas_finales, "n_parametros": int(m["n_parametros"]),
            "perdida_val_media": float(m["perdida_val_media"]),
            "segundos_ajuste": float(tuning[tuning.red == red].segundos.sum()),
            "segundos_final": float(np.sum(segundos)),
            "columnas_entrada": list(X.columns), "horizontes": horizontes,
            "ventana": cfg["preprocesado"]["ventana"],
            "train_inicio": str(cal[final.pos_train[0]].date()),
            "train_fin": str(cal[final.pos_train[-1]].date()),
            "test_inicio": str(cal[final.pos_eval[0]].date()),
            "n_train": int(len(final.pos_train)),
            "rapido": args.rapido,
        }
        with open(dirs["models"] / f"{red}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    pd.DataFrame(filas_final).to_csv(dirs["results"] / "entrenamiento_final.csv", index=False)
    log.info("Mejores configuraciones:\n%s", mejores[["config", "perdida_val_media", "epoca_mediana",
                                                       "n_parametros"]].to_string())
    log.info("Entrenamiento completo en %.1f min", (time.perf_counter() - t_ajuste) / 60)


if __name__ == "__main__":
    main()

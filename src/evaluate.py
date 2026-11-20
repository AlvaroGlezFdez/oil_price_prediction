"""
Intervalos de incertidumbre y evaluación en test (Fase 2).

1. Intervalo del 80 %: para cada red y horizonte, cuantiles 10 y 90 de los residuos
   out-of-fold (y real - y predicha, en log-rentabilidad). Para el ingenuo (y_hat = 0) los
   residuos son la propia y real en los mismos años de validación.  -> results/intervalos.json
2. Test (t >= 2023-01-01, usado una sola vez): métricas por red (media y desviación entre
   las 3 semillas) y por horizonte, más el ingenuo.
   -> results/comparativa.csv, results/test_predictions.csv
3. Gráficos de predicción vs real con banda del 80 % (mejor red según validación e ingenuo).
   -> reports/results/prediccion_h<h>.png

Uso:  python -m src.evaluate [--rapido]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from src.config import cargar_config
from src.models import RedRecurrente, predecir
from src.train import directorios, preparar_datos
from src.windows import Preprocesador, construir_ventanas

AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, TINTA_MUTED, REJILLA, SUPERFICIE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------
# Intervalos
# ---------------------------------------------------------------
def calcular_intervalos(oof: pd.DataFrame, q_inf: float, q_sup: float) -> dict:
    """Cuantiles de los residuos out-of-fold por modelo y horizonte."""
    res = {}
    for red, g in oof.groupby("red"):
        res[red] = {}
        for h, gh in g.groupby("horizonte"):
            e = gh["y_real"] - gh["y_pred"]
            res[red][str(h)] = {"q_inf": float(e.quantile(q_inf)), "q_sup": float(e.quantile(q_sup)),
                                "n_residuos": int(e.notna().sum())}
    # Ingenuo: mismas fechas de validación (se toman de la primera red)
    primera = oof[oof["red"] == oof["red"].iloc[0]]
    res["ingenuo"] = {}
    for h, gh in primera.groupby("horizonte"):
        e = gh["y_real"]
        res["ingenuo"][str(h)] = {"q_inf": float(e.quantile(q_inf)), "q_sup": float(e.quantile(q_sup)),
                                  "n_residuos": int(e.notna().sum())}
    res["_cuantiles"] = [q_inf, q_sup]
    return res


# ---------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------
def metricas(y: np.ndarray, y_hat: np.ndarray, p_t: np.ndarray, q: dict) -> dict:
    """Métricas de un horizonte (y, y_hat en log-rentabilidad; p_t precio en t)."""
    ok = ~np.isnan(y)
    y, y_hat, p_t = y[ok], y_hat[ok], p_t[ok]
    p_real, p_pred = p_t * np.exp(y), p_t * np.exp(y_hat)
    err_usd, err_log = p_pred - p_real, y_hat - y
    dentro = (y >= y_hat + q["q_inf"]) & (y <= y_hat + q["q_sup"])
    return {
        "n_test": int(ok.sum()),
        "sesgo_usd": float(np.mean(err_usd)),
        "MSE_usd": float(np.mean(err_usd ** 2)), "MAE_usd": float(np.mean(np.abs(err_usd))),
        "RMSE_usd": float(np.sqrt(np.mean(err_usd ** 2))),
        "MSE_log": float(np.mean(err_log ** 2)), "MAE_log": float(np.mean(np.abs(err_log))),
        "RMSE_log": float(np.sqrt(np.mean(err_log ** 2))),
        "acierto_direccion_%": float(100 * np.mean(np.sign(y_hat) == np.sign(y))),
        "cobertura_80_%": float(100 * np.mean(dentro)),
        "pct_dias_subida": float(100 * np.mean(y > 0)),
    }


def diebold_mariano(e_modelo: np.ndarray, e_ref: np.ndarray, h: int) -> tuple[float, float]:
    """
    Contraste de Diebold-Mariano (bilateral) con pérdida cuadrática.

    d_t = e_modelo_t^2 - e_ref_t^2. Como las previsiones a h días se solapan, d_t está
    autocorrelacionada hasta el retardo h-1: la varianza de largo plazo se estima con
    Newey-West (núcleo de Bartlett) con h-1 retardos. DM < 0 indica que el modelo tiene
    menor pérdida que la referencia. Devuelve (estadístico, p-valor).
    """
    ok = ~(np.isnan(e_modelo) | np.isnan(e_ref))
    d = e_modelo[ok] ** 2 - e_ref[ok] ** 2
    n = len(d)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * np.dot(dc[k:], dc[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    estad = d.mean() / np.sqrt(lrv / n)
    return float(estad), float(2 * (1 - norm.cdf(abs(estad))))


# ---------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------
def grafico_prediccion(pred: pd.DataFrame, h: int, mejor: str, n_semillas: int, destino) -> None:
    """Precio real en t+h frente al previsto, con banda del 80 %, para la mejor red y el ingenuo."""
    fig, ejes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True, facecolor=SUPERFICIE)
    for ax, (modelo, color, etiqueta) in zip(
            ejes, [(mejor, AZUL, f"{mejor.upper()} (media de {n_semillas} semillas)"), ("ingenuo", NARANJA, "Ingenuo")]):
        g = pred[(pred.modelo == modelo) & (pred.semilla == "media") & (pred.horizonte == h)]
        g = g.dropna(subset=["precio_real"]).sort_values("fecha_objetivo")
        ax.set_facecolor(SUPERFICIE)
        ax.fill_between(g.fecha_objetivo, g.precio_li80, g.precio_ls80, color=color, alpha=0.18, lw=0,
                        label="intervalo 80 %")
        ax.plot(g.fecha_objetivo, g.precio_real, color=TINTA, lw=1.4, label="Brent real")
        ax.plot(g.fecha_objetivo, g.precio_pred, color=color, lw=1.6, label="previsión")
        ax.set_title(f"{etiqueta}: previsión a {h} días hábiles", loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("USD / barril", color=TINTA_2)
        ax.grid(color=REJILLA, lw=0.6)
        ax.tick_params(colors=TINTA_MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    ejes[-1].set_xlabel("fecha objetivo (t + h)", color=TINTA_2)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()

    cfg = cargar_config()
    dirs = directorios(cfg, args.rapido)
    horizontes = cfg["preprocesado"]["horizontes"]
    v = cfg["preprocesado"]["ventana"]
    ci = cfg["intervalo"]

    datos, X, Y, pos, particiones = preparar_datos(cfg)
    final = particiones[-1]
    cal = datos.index

    # 1) Intervalos a partir de los residuos out-of-fold
    oof = pd.read_csv(dirs["results"] / "oof_predictions.csv", parse_dates=["fecha"])
    intervalos = calcular_intervalos(oof, ci["q_inf"], ci["q_sup"])
    with open(dirs["results"] / "intervalos.json", "w", encoding="utf-8") as f:
        json.dump(intervalos, f, indent=2)

    # 2) Predicciones de test con los modelos finales guardados
    with open(dirs["models"] / "preprocesado.json", encoding="utf-8") as f:
        prep = Preprocesador.desde_dict(json.load(f))
    X_te = construir_ventanas(prep.transformar_X(X), final.pos_eval, v)
    Y_te = Y.to_numpy()[final.pos_eval]
    p_t = datos["brent"].to_numpy()[final.pos_eval]
    fechas = cal[final.pos_eval]
    # Fecha objetivo t+h en el calendario (NaT si cae fuera de la muestra)
    fechas_obj = {h: pd.DatetimeIndex([cal[i + h] if i + h < len(cal) else pd.NaT for i in final.pos_eval])
                  for h in horizontes}

    predicciones = {}   # (modelo, semilla) -> matriz (n_test, 3) en log-rentabilidad
    metas = {}
    for red in cfg["entrenamiento"]["redes"]:
        with open(dirs["models"] / f"{red}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        metas[red] = meta
        for s in meta["semillas"]:
            modelo = RedRecurrente(red, X.shape[1], meta["units"], meta["dropout"])
            modelo.load_state_dict(torch.load(dirs["models"] / f"{red}_s{s}.pt", weights_only=True))
            predicciones[(red, str(s))] = prep.desestandarizar_y(predecir(modelo, X_te))
        predicciones[(red, "media")] = np.mean([predicciones[(red, str(s))] for s in meta["semillas"]], axis=0)
    predicciones[("ingenuo", "media")] = np.zeros_like(Y_te)

    # 3) Métricas por modelo, semilla y horizonte
    filas = []
    for (modelo, semilla), y_hat in predicciones.items():
        if semilla == "media" and modelo != "ingenuo":
            continue   # las métricas de las redes se calculan por semilla y luego se promedian
        for j, h in enumerate(horizontes):
            m = metricas(Y_te[:, j], y_hat[:, j], p_t, intervalos[modelo][str(h)])
            filas.append({"modelo": modelo, "semilla": semilla, "horizonte": h, **m})
    por_semilla = pd.DataFrame(filas)
    ingenuo = por_semilla[por_semilla.modelo == "ingenuo"].set_index("horizonte")
    por_semilla["mejora_vs_ingenuo"] = 1 - por_semilla["RMSE_usd"] / por_semilla["horizonte"].map(
        ingenuo["RMSE_usd"])
    por_semilla.to_csv(dirs["results"] / "comparativa_por_semilla.csv", index=False)

    cols = ["sesgo_usd", "MSE_usd", "MAE_usd", "RMSE_usd", "MSE_log", "MAE_log", "RMSE_log", "mejora_vs_ingenuo",
            "acierto_direccion_%", "cobertura_80_%"]
    agg = por_semilla.groupby(["modelo", "horizonte"])[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{'media' if e == 'mean' else 'desv'}" for c, e in agg.columns]
    agg = agg.reset_index()
    base = por_semilla.groupby(["modelo", "horizonte"])[["n_test", "pct_dias_subida"]].first().reset_index()
    comp = base.merge(agg, on=["modelo", "horizonte"])
    # En el ingenuo y_hat = 0: el "acierto de dirección" no aplica; se da el % de subidas
    comp.loc[comp.modelo == "ingenuo", ["acierto_direccion_%_media", "acierto_direccion_%_desv"]] = np.nan
    info = {red: (m["config"], m["n_parametros"], m["epocas_finales"], m["segundos_ajuste"], m["segundos_final"])
            for red, m in metas.items()}
    info["ingenuo"] = ("-", 0, 0, 0.0, 0.0)
    for k, c in enumerate(["config", "n_parametros", "epocas_finales", "segundos_ajuste", "segundos_final"]):
        comp[c] = comp.modelo.map({r: t[k] for r, t in info.items()})
    orden = {m: i for i, m in enumerate(cfg["entrenamiento"]["redes"] + ["ingenuo"])}
    comp = comp.sort_values(["horizonte", "modelo"], key=lambda s: s.map(orden) if s.name == "modelo" else s)
    # Diebold-Mariano de cada red (media de semillas) frente al ingenuo, en dólares
    dm = {}
    for red in cfg["entrenamiento"]["redes"]:
        for j, h in enumerate(horizontes):
            p_real = p_t * np.exp(Y_te[:, j])
            e_red = p_t * np.exp(predicciones[(red, "media")][:, j]) - p_real
            e_ing = p_t - p_real
            dm[(red, h)] = diebold_mariano(e_red, e_ing, h)
    # Métricas de la previsión media de las semillas (la que usa la aplicación)
    ens = {}
    for (modelo, semilla), y_hat in predicciones.items():
        if semilla != "media":
            continue
        for j, h in enumerate(horizontes):
            ens[(modelo, h)] = metricas(Y_te[:, j], y_hat[:, j], p_t, intervalos[modelo][str(h)])
    for c in ["MAE_usd", "RMSE_usd", "sesgo_usd", "acierto_direccion_%", "cobertura_80_%"]:
        comp[f"{c}_ensemble"] = [ens[(m, h)][c] for m, h in zip(comp.modelo, comp.horizonte)]
    rmse_ing = {h: ens[("ingenuo", h)]["RMSE_usd"] for h in horizontes}
    comp["mejora_vs_ingenuo_ensemble"] = 1 - comp["RMSE_usd_ensemble"] / comp.horizonte.map(rmse_ing)
    comp.loc[comp.modelo == "ingenuo", "acierto_direccion_%_ensemble"] = np.nan
    comp["DM_estadistico"] = [dm.get((m, h), (np.nan, np.nan))[0] for m, h in zip(comp.modelo, comp.horizonte)]
    comp["DM_p_valor"] = [dm.get((m, h), (np.nan, np.nan))[1] for m, h in zip(comp.modelo, comp.horizonte)]
    comp.to_csv(dirs["results"] / "comparativa.csv", index=False)

    # 4) Predicciones de test en formato largo (incluye la media de semillas e intervalos)
    largas = []
    for (modelo, semilla), y_hat in predicciones.items():
        for j, h in enumerate(horizontes):
            q = intervalos[modelo][str(h)]
            largas.append(pd.DataFrame({
                "modelo": modelo, "semilla": semilla, "fecha": fechas, "horizonte": h,
                "fecha_objetivo": fechas_obj[h], "precio_t": p_t,
                "y_real": Y_te[:, j], "y_pred": y_hat[:, j],
                "precio_real": p_t * np.exp(Y_te[:, j]), "precio_pred": p_t * np.exp(y_hat[:, j]),
                "precio_li80": p_t * np.exp(y_hat[:, j] + q["q_inf"]),
                "precio_ls80": p_t * np.exp(y_hat[:, j] + q["q_sup"]),
            }))
    pred = pd.concat(largas, ignore_index=True)
    pred.to_csv(dirs["results"] / "test_predictions.csv", index=False)

    # 5) Gráficos: mejor red según la VALIDACIÓN (no según test)
    mejor = min(metas, key=lambda r: metas[r]["perdida_val_media"])
    for h in horizontes:
        grafico_prediccion(pred, h, mejor, len(metas[mejor]["semillas"]), dirs["figuras"] / f"prediccion_h{h}.png")

    # Resumen por consola
    vista = comp[["modelo", "horizonte", "n_test", "MAE_usd_media", "RMSE_usd_media", "RMSE_usd_desv",
                  "mejora_vs_ingenuo_media", "RMSE_usd_ensemble", "mejora_vs_ingenuo_ensemble", "DM_estadistico", "DM_p_valor", "sesgo_usd_media",
                  "acierto_direccion_%_media", "pct_dias_subida",
                  "cobertura_80_%_media", "n_parametros", "segundos_ajuste", "segundos_final"]]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(vista.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nIntervalos (cuantiles de residuos OOF):")
    for mod, d in intervalos.items():
        if not mod.startswith("_"):
            print(f"  {mod:<8}", {h: (round(v_['q_inf'], 3), round(v_['q_sup'], 3)) for h, v_ in d.items()})
    print(f"\nMejor red según validación: {mejor.upper()}")


if __name__ == "__main__":
    main()

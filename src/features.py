"""
Variables de entrada y objetivo (Fase 2).

Entradas (día t, calculadas solo con información de t y anterior):
  r_<serie>     log-rentabilidad diaria ln(P_t / P_{t-1}) de Brent, WTI, futuro WTI,
                gas (NG=F), cobre, oro, índice dólar, USD/EUR y CNY/USD
  ln_vix        logaritmo del VIX
  tipo_bce      tipo de la facilidad de depósito del BCE (nivel)
  d_tipo_bce    cambio diario del tipo BCE
  vol20_brent   desviación típica de la log-rentabilidad del Brent en t-19..t
  dist_mm252    ln(P_t) - ln(media del precio del Brent en t-251..t)

Objetivo (multisalida):
  y_h(t) = ln(P_{t+h} / P_t), h = 5, 63, 252. NaN cuando t+h está fuera de la muestra.
"""
import numpy as np
import pandas as pd


def construir_entradas(datos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Devuelve un DataFrame (índice = calendario del Brent) con las variables de entrada."""
    cv = cfg["variables"]
    X = pd.DataFrame(index=datos.index)

    # Log-rentabilidades diarias: diff() solo mira al día anterior (pasado)
    for col in cv["rentabilidades"]:
        X[f"r_{col}"] = np.log(datos[col]).diff()

    # Series en logaritmo (nivel)
    for col in cv["log_nivel"]:
        X[f"ln_{col}"] = np.log(datos[col])

    # Series en nivel y en cambio diario
    for col in cv["nivel_y_cambio"]:
        X[col] = datos[col]
        X[f"d_{col}"] = datos[col].diff()

    # Variables propias del Brent (ventanas móviles que terminan en t: sin mirar al futuro)
    r_brent = np.log(datos["brent"]).diff()
    X[f"vol{cv['vol_ventana']}_brent"] = r_brent.rolling(cv["vol_ventana"]).std()
    mm = datos["brent"].rolling(cv["media_movil"]).mean()
    X[f"dist_mm{cv['media_movil']}"] = np.log(datos["brent"]) - np.log(mm)

    return X


def construir_objetivo(brent: pd.Series, horizontes: list[int]) -> pd.DataFrame:
    """y_h(t) = ln(P_{t+h} / P_t) sobre el calendario del Brent (h en días hábiles)."""
    ln_p = np.log(brent)
    return pd.DataFrame({f"y_{h}": ln_p.shift(-h) - ln_p for h in horizontes}, index=brent.index)

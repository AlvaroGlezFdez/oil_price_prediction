"""
Descarga y alineación de datos (Fase 1).

Basado en scripts/descarga_datos_petroleo.py (script del profesor) con una
corrección del calendario:

  * En el script original se hace ``ffill()`` sobre la unión de TODAS las fechas
    y DESPUÉS se filtra por ``brent`` no nulo. Como el ffill ya ha rellenado el
    Brent, ese filtro no elimina nada: aparecen "días falsos" del Brent (festivos
    en Londres en los que Yahoo/FRED-EE. UU. sí cotizan) con el precio del día
    anterior repetido, lo que genera rentabilidades nulas ficticias.

  * Aquí el calendario se define PRIMERO como los días con dato real del Brent en
    FRED. El resto de series se rellenan solo hacia delante (nunca hacia atrás)
    usando su propia historia y se reindexan a ese calendario.

Salidas:
  data/raw/fred_<ID>.csv, data/raw/yahoo_<ticker>.csv  -> descargas sin tocar
  data/processed/dataset_sin_relleno.csv               -> alineado, sin ffill (para el EDA)
  data/processed/dataset.csv                           -> alineado y con ffill (precios en niveles)

Uso:  python -m src.download
"""
from datetime import datetime
from io import StringIO
import time

import numpy as np
import pandas as pd
import requests

from src.config import cargar_config, ruta


# ---------------------------------------------------------------
# Utilidades de descarga
# ---------------------------------------------------------------
def _con_reintentos(funcion, reintentos: int, descripcion: str):
    """Ejecuta `funcion` reintentando con espera creciente si falla."""
    for intento in range(1, reintentos + 1):
        try:
            return funcion()
        except Exception as e:  # noqa: BLE001 - queremos reintentar ante cualquier fallo de red
            if intento == reintentos:
                raise RuntimeError(f"Fallo descargando {descripcion}: {e}") from e
            espera = 5 * intento
            print(f"  aviso: {descripcion} falló ({e}); reintento en {espera} s")
            time.sleep(espera)


def descargar_fred(serie: str, nombre: str, inicio: str, fin: str, cfg_datos: dict) -> pd.Series:
    """Descarga una serie de FRED (endpoint CSV público, sin clave)."""
    url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={serie}&cosd={inicio}&coed={fin}")

    def _bajar():
        r = requests.get(url, timeout=cfg_datos["timeout_s"])
        r.raise_for_status()
        return r.text

    texto = _con_reintentos(_bajar, cfg_datos["reintentos"], f"FRED {serie}")
    df = pd.read_csv(StringIO(texto), index_col=0, parse_dates=True, na_values=".")
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(nombre)
    s.index.name = "Date"
    return s


def _normalizar_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana columnas MultiIndex (yfinance reciente) y quita la zona horaria del índice."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df


def descargar_yahoo_ohlcv(ticker: str, prefijo: str, inicio: str, fin: str, cfg_datos: dict) -> pd.DataFrame:
    """Descarga OHLCV diario de un ticker de Yahoo Finance."""
    import yfinance as yf

    def _bajar():
        df = yf.download(ticker, start=inicio, end=fin, auto_adjust=False,
                         progress=False, multi_level_index=False)
        if df is None or df.empty:
            raise ValueError("respuesta vacía")
        return df

    df = _normalizar_yahoo(_con_reintentos(_bajar, cfg_datos["reintentos"], f"Yahoo {ticker}"))
    return df[["Open", "High", "Low", "Close", "Volume"]].add_prefix(f"{prefijo}_")


def descargar_yahoo_cierre(ticker: str, nombre: str, inicio: str, fin: str, cfg_datos: dict) -> pd.Series:
    """Descarga solo el cierre diario de un ticker de Yahoo Finance."""
    return descargar_yahoo_ohlcv(ticker, "x", inicio, fin, cfg_datos)["x_Close"].rename(nombre)


# ---------------------------------------------------------------
# Alineación al calendario del Brent
# ---------------------------------------------------------------
def calendario_brent(brent: pd.Series, inicio: str) -> pd.DatetimeIndex:
    """Días con dato REAL del Brent en FRED (laborables y desde `inicio`)."""
    dias = brent.dropna().index
    dias = dias[(dias.day_of_week < 5) & (dias >= pd.Timestamp(inicio))]
    return pd.DatetimeIndex(dias.sort_values().unique(), name="Date")


def alinear(serie: pd.Series, calendario: pd.DatetimeIndex, rellenar: bool, limite=None) -> pd.Series:
    """
    Lleva una serie a `calendario`.

    Si `rellenar` es True, primero se rellena hacia delante sobre la unión de sus
    propias fechas y las del calendario (así un dato publicado en un día que no es
    del Brent se propaga al siguiente día del Brent) y luego se reindexa. Solo se
    usa información pasada o del mismo día: nunca se rellena hacia atrás.
    """
    serie = serie[~serie.index.duplicated(keep="last")].sort_index()
    if not rellenar:
        return serie.reindex(calendario)
    union = serie.index.union(calendario)
    return serie.reindex(union).ffill(limit=limite).reindex(calendario)


# ---------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------
def main() -> None:
    cfg = cargar_config()
    cd = cfg["datos"]
    inicio = cd["inicio"]
    fin = cd["fin"] or datetime.now().strftime("%Y-%m-%d")
    dir_raw, dir_proc = ruta(cfg, "raw"), ruta(cfg, "processed")

    series: dict[str, pd.Series] = {}

    # 1) FRED -----------------------------------------------------
    print("Descargando FRED...")
    for serie, nombre in cd["series_fred"].items():
        s = descargar_fred(serie, nombre, inicio, fin, cd)
        s.to_csv(dir_raw / f"fred_{serie}.csv")
        series[nombre] = s
        print(f"  {serie:<13} -> {nombre:<12} {s.notna().sum():>6} obs. "
              f"({s.first_valid_index():%Y-%m-%d} a {s.last_valid_index():%Y-%m-%d})")

    # 2) Yahoo Finance -------------------------------------------
    # yfinance trata `end` como exclusivo: se suma un día para incluir hoy.
    fin_yahoo = (pd.Timestamp(fin) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print("Descargando Yahoo Finance...")
    for ticker, prefijo in cd["yahoo_ohlcv"].items():
        df = descargar_yahoo_ohlcv(ticker, prefijo, inicio, fin_yahoo, cd)
        df.to_csv(dir_raw / f"yahoo_{ticker.replace('=', '_')}.csv")
        for col in df.columns:
            series[col] = df[col]
        c = df[f"{prefijo}_Close"]
        print(f"  {ticker:<13} -> {prefijo:<12} {c.notna().sum():>6} obs. "
              f"({c.first_valid_index():%Y-%m-%d} a {c.last_valid_index():%Y-%m-%d})")
    for ticker, nombre in cd["yahoo_cierres"].items():
        s = descargar_yahoo_cierre(ticker, nombre, inicio, fin_yahoo, cd)
        s.to_frame().to_csv(dir_raw / f"yahoo_{ticker.replace('=', '_')}.csv")
        series[nombre] = s
        print(f"  {ticker:<13} -> {nombre:<12} {s.notna().sum():>6} obs. "
              f"({s.first_valid_index():%Y-%m-%d} a {s.last_valid_index():%Y-%m-%d})")

    # 3) WTI <= 0 -> faltante (antes de alinear y rellenar) ------
    for col in cd["columnas_wti"]:
        if col in series:
            malos = series[col] <= 0
            if malos.any():
                fechas = ", ".join(f"{d:%Y-%m-%d}" for d in series[col].index[malos])
                print(f"  {col}: {malos.sum()} valor(es) <= 0 marcados como faltantes ({fechas})")
                series[col] = series[col].mask(malos)

    # 4) Calendario del Brent y alineación ----------------------
    cal = calendario_brent(series["brent"], inicio)
    sin_relleno = pd.DataFrame({k: alinear(v, cal, rellenar=False) for k, v in series.items()})
    con_relleno = pd.DataFrame({k: alinear(v, cal, rellenar=True, limite=cd["ffill_limite"])
                                for k, v in series.items()})

    # Comprobaciones de coherencia
    assert con_relleno["brent"].notna().all(), "El Brent no puede tener huecos en su propio calendario"
    assert (con_relleno["brent"] == sin_relleno["brent"]).all(), "El Brent no debe rellenarse nunca"
    assert np.isfinite(con_relleno["brent"]).all() and (con_relleno["brent"] > 0).all()

    sin_relleno.to_csv(dir_proc / "dataset_sin_relleno.csv")
    con_relleno.to_csv(dir_proc / "dataset.csv")

    print(f"\nCalendario del Brent: {len(cal)} días "
          f"({cal[0]:%Y-%m-%d} a {cal[-1]:%Y-%m-%d}), {con_relleno.shape[1]} columnas")
    print(f"Guardado en {dir_proc / 'dataset.csv'}")


if __name__ == "__main__":
    main()

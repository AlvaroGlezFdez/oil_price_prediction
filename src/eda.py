"""
Análisis exploratorio de datos (Fase 1).

Lee data/processed/dataset.csv (con relleno hacia delante) y
data/processed/dataset_sin_relleno.csv (alineado al calendario del Brent, sin relleno)
y genera en reports/eda/:

  brent_precio_y_rentabilidad.png   Brent (precio) y su log-rentabilidad diaria
  estacionariedad_adf_kpss.csv      ADF y KPSS sobre ln(precio) y log-rentabilidad del Brent
  faltantes.csv                     % de faltantes por columna antes y después del relleno
  estadisticos_niveles.csv          estadísticos básicos de las series en niveles
  estadisticos_rentabilidades.csv   estadísticos básicos de las log-rentabilidades
  brent_extremos.csv                15 días con |log-rentabilidad| del Brent más extrema
  correlaciones_rentabilidades.csv  matriz de correlaciones entre log-rentabilidades
  correlaciones_rentabilidades.png  la misma matriz como mapa de calor
  resumen_eda.md                    resumen legible de todo lo anterior

Uso:  python -m src.eda
"""
import warnings

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo se guardan ficheros
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import cargar_config, fijar_semillas, ruta

# ---------------------------------------------------------------
# Estilo de los gráficos (paleta neutra y validada)
# ---------------------------------------------------------------
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_MUTED = "#898781"
REJILLA = "#e4e3df"
AZUL = "#2a78d6"          # serie principal
BANDA = "#f0efec"         # sombreado de episodios
# Divergente azul <-> rojo con punto medio gris (correlación -1 ... +1)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#9f2526"])

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
    "axes.edgecolor": TINTA_MUTED, "axes.labelcolor": TINTA_2,
    "xtick.color": TINTA_MUTED, "ytick.color": TINTA_MUTED,
    "text.color": TINTA, "axes.titlecolor": TINTA,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "savefig.dpi": 150, "savefig.bbox": "tight",
})

# Episodios históricos que se sombrean en el gráfico del Brent
EPISODIOS = [
    ("2008-01-01", "2009-03-31", "Crisis 2008"),
    ("2014-06-01", "2016-02-29", "Caída 2014-16"),
    ("2020-02-15", "2020-06-30", "COVID-19"),
    ("2022-02-24", "2022-08-31", "Invasión de Ucrania"),
]

# Series de precio para las que se calcula la log-rentabilidad
COLUMNAS_PRECIO = ["brent", "wti", "brent_fut_Close", "wti_fut_Close", "gas_natural",
                   "cobre", "oro", "indice_dolar", "usd_por_eur", "cny_por_usd"]


# ---------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------
def tabla_md(df: pd.DataFrame, decimales: int = 4, indice: bool = True) -> str:
    """Convierte un DataFrame en una tabla Markdown (sin dependencias extra)."""
    d = df.reset_index() if indice else df.copy()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{decimales}f}"
        if isinstance(v, pd.Timestamp):
            return f"{v:%Y-%m-%d}"
        return str(v)

    cab = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * d.shape[1]) + "|"
    filas = ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in d.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def log_rentabilidades(datos: pd.DataFrame) -> pd.DataFrame:
    """Log-rentabilidades diarias de las series de precio y log-cambio del VIX."""
    cols = [c for c in COLUMNAS_PRECIO if c in datos.columns]
    r = np.log(datos[cols]).diff()
    if "vix" in datos.columns:
        r["vix_dlog"] = np.log(datos["vix"]).diff()
    return r


def racha_maxima(faltante: pd.Series) -> int:
    """Longitud de la racha más larga de valores True consecutivos."""
    if not faltante.any():
        return 0
    grupos = (faltante != faltante.shift()).cumsum()
    return int(faltante.groupby(grupos).sum().max())


# ---------------------------------------------------------------
# Bloques del EDA
# ---------------------------------------------------------------
def grafico_brent(datos: pd.DataFrame, r_brent: pd.Series, destino) -> None:
    """Precio del Brent y log-rentabilidad diaria, con episodios sombreados."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for ax in (ax1, ax2):
        for ini, fin, _ in EPISODIOS:
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=BANDA, zorder=0, lw=0)

    ax1.plot(datos.index, datos["brent"], color=AZUL, lw=1.2)
    ax1.set_title("Brent spot (FRED DCOILBRENTEU)", loc="left")
    ax1.set_ylabel("USD / barril")
    ymax = datos["brent"].max()
    # Etiquetas a dos alturas alternas para que no se solapen
    for k, (ini, fin, nombre) in enumerate(EPISODIOS):
        centro = pd.Timestamp(ini) + (pd.Timestamp(fin) - pd.Timestamp(ini)) / 2
        altura = ymax * (1.03 if k % 2 == 0 else 1.10)
        ax1.text(centro, altura, nombre, ha="center", va="bottom", fontsize=8, color=TINTA_2)
    ax1.set_ylim(0, ymax * 1.18)

    # Máximo histórico anotado
    f_max = datos["brent"].idxmax()
    ax1.annotate(f"Máx. {datos['brent'].max():.1f} $ ({f_max:%d-%m-%Y})",
                 xy=(f_max, datos["brent"].max()), xytext=(15, -25), textcoords="offset points",
                 fontsize=8, color=TINTA_2, arrowprops=dict(arrowstyle="-", color=TINTA_MUTED, lw=0.8))

    ax2.plot(r_brent.index, r_brent, color=AZUL, lw=0.6)
    ax2.axhline(0, color=TINTA_MUTED, lw=0.8)
    ax2.set_title("Log-rentabilidad diaria del Brent", loc="left")
    ax2.set_ylabel("ln(P_t / P_{t-1})")
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def pruebas_estacionariedad(brent: pd.Series, r_brent: pd.Series) -> pd.DataFrame:
    """
    ADF (H0: raíz unitaria) y KPSS (H0: estacionaria) sobre ln(precio) y log-rentabilidad.
    Para ln(precio) se prueba con constante y con constante + tendencia.
    Nota: statsmodels acota el p-valor de KPSS a [0,01; 0,10].
    """
    filas = []
    casos = [("ln(precio Brent)", np.log(brent), "c"),
             ("ln(precio Brent)", np.log(brent), "ct"),
             ("log-rentabilidad Brent", r_brent.dropna(), "c")]
    for nombre, serie, reg in casos:
        adf_est, adf_p, adf_lags, _, adf_crit, _ = adfuller(serie, regression=reg, autolag="AIC", result_object=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_est, kpss_p, kpss_lags, kpss_crit = kpss(serie, regression=reg, nlags="auto", result_object=False)
        filas.append({
            "serie": nombre,
            "especificacion": "constante" if reg == "c" else "constante + tendencia",
            "n": len(serie),
            "ADF_estadistico": adf_est, "ADF_crit_5%": adf_crit["5%"], "ADF_p": adf_p,
            "ADF_retardos": adf_lags,
            "KPSS_estadistico": kpss_est, "KPSS_crit_5%": kpss_crit["5%"], "KPSS_p": kpss_p,
            "KPSS_retardos": kpss_lags,
            "ADF_conclusion_5%": "estacionaria" if adf_p < 0.05 else "raíz unitaria",
            "KPSS_conclusion_5%": "no estacionaria" if kpss_p < 0.05 else "estacionaria",
        })
    return pd.DataFrame(filas)


def tabla_faltantes(sin_relleno: pd.DataFrame, con_relleno: pd.DataFrame) -> pd.DataFrame:
    """% de faltantes por columna antes y después del relleno hacia delante."""
    filas = []
    for col in sin_relleno.columns:
        antes, despues = sin_relleno[col].isna(), con_relleno[col].isna()
        primera = sin_relleno[col].first_valid_index()
        desde = antes.loc[primera:] if primera is not None else antes
        filas.append({
            "columna": col,
            "primer_dato": primera,
            "ultimo_dato_real": sin_relleno[col].last_valid_index(),
            "%_faltante_antes": 100 * antes.mean(),
            "%_faltante_antes_desde_primer_dato": 100 * desde.mean(),
            "%_faltante_despues": 100 * despues.mean(),
            "dias_rellenados": int((antes & ~despues).sum()),
            "racha_max_faltantes_antes": racha_maxima(desde),
        })
    return pd.DataFrame(filas).set_index("columna")


def estadisticos(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos básicos por columna, incluidas asimetría, curtosis y fechas de mín./máx."""
    t = df.describe().T
    t["asimetria"] = df.skew()
    t["curtosis_exceso"] = df.kurt()
    t["fecha_min"] = df.idxmin()
    t["fecha_max"] = df.idxmax()
    return t


def extremos_brent(brent: pd.Series, r_brent: pd.Series, n: int) -> pd.DataFrame:
    """Los n días con |log-rentabilidad| del Brent más extrema."""
    idx = r_brent.abs().nlargest(n).index
    t = pd.DataFrame({
        "precio_anterior": brent.shift(1).loc[idx],
        "precio": brent.loc[idx],
        "log_rentabilidad": r_brent.loc[idx],
        "variacion_%": 100 * (np.exp(r_brent.loc[idx]) - 1),
    })
    t.index.name = "fecha"
    return t


def grafico_correlaciones(corr: pd.DataFrame, destino) -> None:
    """Mapa de calor de la matriz de correlaciones (escala divergente fija en [-1, 1])."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.75 * n + 2.5, 0.75 * n + 1.5))
    im = ax.imshow(corr.values, cmap=CMAP_DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    # Huecos de 2 px entre celdas
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(v) > 0.6 else TINTA)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    ax.set_title("Correlación de Pearson entre log-rentabilidades diarias", loc="left")
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


# ---------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------
def main() -> None:
    cfg = cargar_config()
    fijar_semillas(cfg["semilla"])
    dir_proc, dir_eda = ruta(cfg, "processed"), ruta(cfg, "eda")

    datos = pd.read_csv(dir_proc / "dataset.csv", index_col=0, parse_dates=True)
    sin_relleno = pd.read_csv(dir_proc / "dataset_sin_relleno.csv", index_col=0, parse_dates=True)

    r = log_rentabilidades(datos)
    r_brent = r["brent"]

    # 1) Gráfico del Brent
    grafico_brent(datos, r_brent, dir_eda / "brent_precio_y_rentabilidad.png")

    # 2) Estacionariedad
    est = pruebas_estacionariedad(datos["brent"], r_brent)
    est.to_csv(dir_eda / "estacionariedad_adf_kpss.csv", index=False)

    # 3) Faltantes
    falt = tabla_faltantes(sin_relleno, datos)
    falt.to_csv(dir_eda / "faltantes.csv")

    # 4) Estadísticos y extremos
    cols_niveles = [c for c in COLUMNAS_PRECIO + ["vix", "tipo_bce"] if c in datos.columns]
    est_niv = estadisticos(datos[cols_niveles])
    est_ret = estadisticos(r)
    est_niv.to_csv(dir_eda / "estadisticos_niveles.csv")
    est_ret.to_csv(dir_eda / "estadisticos_rentabilidades.csv")
    ext = extremos_brent(datos["brent"], r_brent, cfg["eda"]["n_extremos"])
    ext.to_csv(dir_eda / "brent_extremos.csv")

    # 5) Correlaciones (por pares: el futuro del Brent solo existe desde ~2007)
    corr = r.corr()
    corr.to_csv(dir_eda / "correlaciones_rentabilidades.csv")
    grafico_correlaciones(corr, dir_eda / "correlaciones_rentabilidades.png")

    # 6) Resumen en Markdown
    cols_est = ["count", "mean", "std", "min", "max", "asimetria", "curtosis_exceso", "fecha_min", "fecha_max"]
    md = [
        "# Análisis exploratorio (Fase 1)",
        "",
        f"- Calendario: días con dato real del Brent en FRED, "
        f"{datos.index[0]:%Y-%m-%d} a {datos.index[-1]:%Y-%m-%d} ({len(datos)} filas, {datos.shape[1]} columnas).",
        "- Relleno: solo hacia delante, nunca hacia atrás. WTI <= 0 marcado como faltante antes de rellenar.",
        "",
        "![Brent](brent_precio_y_rentabilidad.png)",
        "",
        "## Estacionariedad (ADF: H0 raíz unitaria; KPSS: H0 estacionaria)",
        "",
        tabla_md(est[["serie", "especificacion", "n", "ADF_estadistico", "ADF_p",
                      "KPSS_estadistico", "KPSS_p", "ADF_conclusion_5%", "KPSS_conclusion_5%"]],
                 indice=False),
        "",
        "El p-valor de KPSS está acotado por statsmodels a [0,01; 0,10].",
        "",
        "## Faltantes por columna (sobre el calendario del Brent)",
        "",
        tabla_md(falt, decimales=2),
        "",
        "## Estadísticos de las log-rentabilidades diarias",
        "",
        tabla_md(est_ret[cols_est], decimales=4),
        "",
        "## Estadísticos de las series en niveles",
        "",
        tabla_md(est_niv[cols_est], decimales=2),
        "",
        f"## {cfg['eda']['n_extremos']} días con |log-rentabilidad| del Brent más extrema",
        "",
        tabla_md(ext, decimales=4),
        "",
        "## Correlaciones entre log-rentabilidades",
        "",
        "![Correlaciones](correlaciones_rentabilidades.png)",
        "",
        tabla_md(corr, decimales=2),
        "",
    ]
    (dir_eda / "resumen_eda.md").write_text("\n".join(md), encoding="utf-8")

    # Salida por consola
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("ESTACIONARIEDAD\n", est[["serie", "especificacion", "ADF_estadistico", "ADF_p",
                                    "KPSS_estadistico", "KPSS_p"]].round(4).to_string(index=False))
    print("\nFALTANTES\n", falt.to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nEXTREMOS BRENT\n", ext.round(4).to_string())
    print("\nESTADÍSTICOS RENTABILIDADES\n", est_ret[cols_est].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nCORRELACIONES\n", corr.round(2).to_string())
    print(f"\nResultados guardados en {dir_eda}")


if __name__ == "__main__":
    main()

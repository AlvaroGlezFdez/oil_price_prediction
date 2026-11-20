"""
Exporta a web/data/ los JSON que lee la web estática (la web no calcula nada).

  historico.json    precio diario del Brent desde 2000 (fechas y precios)
  prediccion.json   predicción actual (copia de results/prediccion_actual.json)
  comparativa.json  results/comparativa.csv (incluye Diebold-Mariano y sesgo)
  test.json         test 2023-2026: real vs RNN (media de semillas) vs ingenuo, por horizonte
  ciclos.json       ciclos históricos con sus extremos calculados con los propios datos
  meta.json         fechas del último dato, del entrenamiento y de la generación

Uso:  python -m src.exportar_web
"""
import json
import math
from datetime import datetime, timezone

import pandas as pd

from src.config import RAIZ, cargar_config

REPO = "AlvaroGlezFdez/oil_price_prediction"

# Ciclos: ventana para sombrear y buscar extremos, y texto breve. Los extremos ({max}, {min},
# fechas) se rellenan con los datos.
CICLOS = [
    {
        "id": "2008", "titulo": "Máximo histórico y crisis financiera",
        "inicio": "2008-01-01", "fin": "2009-03-31",
        "texto": ("El Brent marcó su máximo histórico el {f_max} ({max} $) por la fuerte demanda de los "
                  "emergentes y la especulación. La crisis financiera hundió la demanda y el precio cayó "
                  "a {min} $ el {f_min}: más de un 75 % en menos de seis meses."),
    },
    {
        "id": "2014", "titulo": "Exceso de oferta 2014-2016",
        "inicio": "2014-06-01", "fin": "2016-02-29",
        "texto": ("El auge del petróleo de esquisto en EE. UU. y la decisión de la OPEP de no recortar "
                  "producción (noviembre de 2014) llevaron el precio de {max} $ ({f_max}) a {min} $ "
                  "({f_min})."),
    },
    {
        "id": "2020", "titulo": "Pandemia de COVID-19",
        "inicio": "2020-02-15", "fin": "2020-06-30",
        "texto": ("Los confinamientos desplomaron la demanda. El Brent tocó {min} $ el {f_min} y el WTI "
                  "llegó a cotizar en negativo ({wti_min} $ el 20-04-2020) porque no quedaba espacio "
                  "para almacenar crudo en Cushing."),
    },
    {
        "id": "2022", "titulo": "Invasión de Ucrania",
        "inicio": "2022-02-24", "fin": "2022-08-31",
        "texto": ("Tras la invasión rusa de Ucrania y las sanciones a Rusia, el Brent subió hasta {max} $ "
                  "el {f_max} y después fue bajando a medida que se reorganizaron los flujos de crudo."),
    },
    {
        "id": "2025", "titulo": "Movimientos recientes 2025-2026",
        "inicio": "2025-01-01", "fin": None,   # hasta el último dato
        "texto": ("El Brent bajó durante 2025 hasta {min} $ ({f_min}). En 2026 hubo dos picos. "
                  "Abril ({max} $ el {f_max}, desde unos 61 $ en enero): conflicto entre EE. UU. e Irán y "
                  "cierre del estrecho de Ormuz; el Brent físico llegó a cotizar más de 25 $ por encima del "
                  "futuro por la escasez inmediata de suministro. Octubre ({max2} $ el {f_max2}): nuevas "
                  "tensiones en Oriente Medio (ataques de los hutíes contra Arabia Saudí y despliegue militar "
                  "de EE. UU. en la zona) y ataques entre Rusia y Ucrania a infraestructuras energéticas. "
                  "Entre ambos picos el precio cayó hasta unos 70 $ en junio. Último dato: {ultimo} $ "
                  "({f_ultimo})."),
        "fuentes": ("EIA (Today in Energy, abril de 2026), Euronews (30-04-2026), "
                    "World Oil (16-10-2026)."),
    },
]


def _fecha(ts) -> str:
    return pd.Timestamp(ts).strftime("%d-%m-%Y")


def _limpiar(obj):
    """Convierte NaN/inf en None para que el JSON sea válido."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _limpiar(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_limpiar(v) for v in obj]
    return obj


def construir_ciclos(brent: pd.Series, wti_raw: pd.Series) -> list[dict]:
    salida = []
    for c in CICLOS:
        fin = c["fin"] or str(brent.index[-1].date())
        s = brent.loc[c["inicio"]:fin]
        valores = {"max": f"{s.max():.1f}", "f_max": _fecha(s.idxmax()),
                   "min": f"{s.min():.1f}", "f_min": _fecha(s.idxmin()),
                   "ultimo": f"{brent.iloc[-1]:.1f}", "f_ultimo": _fecha(brent.index[-1]),
                   "wti_min": f"{wti_raw.loc['2020-04-20':'2020-04-20'].min():.2f}"}
        extremos = [{"fecha": str(s.idxmax().date()), "precio": float(s.max()), "tipo": "máximo"},
                    {"fecha": str(s.idxmin().date()), "precio": float(s.min()), "tipo": "mínimo"}]
        if c["id"] == "2025":
            # Mínimo de 2025 y los dos picos de 2026 (antes y después de junio)
            s25 = brent.loc["2025-01-01":"2025-12-31"]
            p1 = brent.loc["2026-01-01":"2026-06-30"]
            p2 = brent.loc["2026-07-01":fin]
            valores.update({"min": f"{s25.min():.1f}", "f_min": _fecha(s25.idxmin()),
                            "max": f"{p1.max():.1f}", "f_max": _fecha(p1.idxmax()),
                            "max2": f"{p2.max():.1f}", "f_max2": _fecha(p2.idxmax())})
            extremos = [{"fecha": str(s25.idxmin().date()), "precio": float(s25.min()), "tipo": "mínimo"},
                        {"fecha": str(p1.idxmax().date()), "precio": float(p1.max()), "tipo": "máximo"},
                        {"fecha": str(p2.idxmax().date()), "precio": float(p2.max()), "tipo": "máximo"}]
        salida.append({"id": c["id"], "titulo": c["titulo"], "inicio": c["inicio"], "fin": fin,
                       # coma decimal (las fechas usan guiones, no se ven afectadas)
                       "texto": c["texto"].format(**{k: v.replace(".", ",") for k, v in valores.items()}), "extremos": extremos,
                       "todo": c.get("todo"), "fuentes": c.get("fuentes")})
    return salida


def main() -> None:
    cfg = cargar_config()
    res = RAIZ / cfg["rutas"]["results"]
    destino = RAIZ / "web" / "data"
    destino.mkdir(parents=True, exist_ok=True)

    def guardar(nombre: str, obj) -> None:
        (destino / nombre).write_text(json.dumps(_limpiar(obj), ensure_ascii=False, separators=(",", ":")),
                                      encoding="utf-8")

    datos = pd.read_csv(RAIZ / cfg["rutas"]["processed"] / "dataset.csv", index_col=0, parse_dates=True)
    brent = datos["brent"]
    guardar("historico.json", {"fechas": [d.strftime("%Y-%m-%d") for d in brent.index],
                               "precios": [round(float(p), 2) for p in brent]})

    pred = json.loads((res / "prediccion_actual.json").read_text(encoding="utf-8"))
    guardar("prediccion.json", pred)

    comp = pd.read_csv(res / "comparativa.csv")
    guardar("comparativa.json", comp.to_dict(orient="records"))

    # Test: RNN de producción elegida (media de semillas) e ingenuo
    red = cfg["produccion"]["red"]
    tp = pd.read_csv(res / "test_predictions.csv", parse_dates=["fecha", "fecha_objetivo"])
    tp = tp[(tp.semilla.astype(str) == "media") & tp.modelo.isin([red, "ingenuo"])].dropna(subset=["precio_real"])
    test = {"red": red, "horizontes": {}}
    for h, g in tp.groupby("horizonte"):
        g = g.sort_values("fecha_objetivo")
        r, i = g[g.modelo == red], g[g.modelo == "ingenuo"]
        test["horizontes"][str(h)] = {
            "fechas": [d.strftime("%Y-%m-%d") for d in r.fecha_objetivo],
            "real": r.precio_real.round(2).tolist(),
            "red": r.precio_pred.round(2).tolist(),
            "red_li": r.precio_li80.round(2).tolist(), "red_ls": r.precio_ls80.round(2).tolist(),
            "ingenuo": i.precio_pred.round(2).tolist(),
        }
    guardar("test.json", test)

    wti_raw = pd.read_csv(RAIZ / cfg["rutas"]["raw"] / "fred_DCOILWTICO.csv", index_col=0, parse_dates=True).iloc[:, 0]
    guardar("ciclos.json", construir_ciclos(brent, wti_raw))

    meta_prod = json.loads((RAIZ / cfg["rutas"]["models"] / "produccion" / "meta.json").read_text(encoding="utf-8"))
    guardar("meta.json", {
        "fecha_ultimo_dato": str(brent.index[-1].date()),
        "precio_ultimo": float(brent.iloc[-1]),
        "fecha_entrenamiento": meta_prod["fecha_entrenamiento"],
        "fecha_generacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": pred["modelo"],
        "repo": REPO,
        "url_actions": f"https://github.com/{REPO}/actions",
    })
    print(f"JSON exportados en {destino}: {sorted(p.name for p in destino.glob('*.json'))}")


if __name__ == "__main__":
    main()

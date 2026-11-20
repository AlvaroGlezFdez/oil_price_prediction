"""
Corrección puntual del tiempo de ajuste de la LSTM (se ejecuta una sola vez).

Durante el ajuste, la ejecución LSTM u32_d0.0 / pliegue_2020 registró 10.275,7 s porque el PC
se suspendió (el log salta de 19:57 a 22:48). Se repitió de forma aislada con la misma semilla:
dio exactamente el mismo resultado (pérdida 4,9331, época óptima 11 de 21) en 15,6 s, ya que el
entrenamiento es determinista. Este script sustituye ese tiempo en results/tuning.csv,
results/tuning_resumen.csv y models/lstm_meta.json. Después hay que volver a ejecutar
`python -m src.evaluate` para regenerar results/comparativa.csv.

Uso:  python -m scripts.corregir_tiempo_lstm
"""
import json

import pandas as pd

from src.config import RAIZ

SEGUNDOS_MEDIDOS = 15.6
FILA = {"red": "lstm", "config": "u32_d0.0", "pliegue": "pliegue_2020"}


def main() -> None:
    ruta_tuning = RAIZ / "results" / "tuning.csv"
    tuning = pd.read_csv(ruta_tuning)
    mask = (tuning.red == FILA["red"]) & (tuning.config == FILA["config"]) & (tuning.pliegue == FILA["pliegue"])
    assert mask.sum() == 1, "No se encuentra la fila afectada"
    antes = float(tuning.loc[mask, "segundos"].iloc[0])
    if antes < 1000:
        print(f"Ya corregido ({antes:.1f} s); no se hace nada.")
        return
    tuning.loc[mask, "segundos"] = SEGUNDOS_MEDIDOS
    tuning.to_csv(ruta_tuning, index=False)

    # Resumen del ajuste: se recalcula la suma de segundos por red y configuración
    ruta_res = RAIZ / "results" / "tuning_resumen.csv"
    resumen = pd.read_csv(ruta_res)
    suma = tuning.groupby(["red", "config"])["segundos"].sum()
    resumen["segundos"] = [suma[(r, c)] for r, c in zip(resumen.red, resumen.config)]
    resumen.to_csv(ruta_res, index=False)

    # Metadatos del modelo final de la LSTM
    ruta_meta = RAIZ / "models" / "lstm_meta.json"
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    meta["segundos_ajuste"] = float(tuning[tuning.red == "lstm"].segundos.sum())
    meta["nota_tiempo"] = (f"Tiempo corregido: la ejecución {FILA['config']}/{FILA['pliegue']} registró "
                           f"{antes:.1f} s por una suspensión del PC; repetida dio el mismo resultado en "
                           f"{SEGUNDOS_MEDIDOS} s.")
    ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Corregido: {antes:.1f} s -> {SEGUNDOS_MEDIDOS} s. "
          f"Tiempo total de ajuste de la LSTM: {meta['segundos_ajuste']:.1f} s")


if __name__ == "__main__":
    main()

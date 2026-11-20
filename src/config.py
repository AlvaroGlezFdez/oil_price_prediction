"""Utilidades comunes: carga de config.yaml, rutas del proyecto y semillas."""
from pathlib import Path
import random

import numpy as np
import yaml

# Raíz del repositorio (carpeta que contiene config.yaml)
RAIZ = Path(__file__).resolve().parent.parent


def cargar_config(ruta: Path | str = RAIZ / "config.yaml") -> dict:
    """Lee config.yaml y devuelve un diccionario."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta(cfg: dict, clave: str) -> Path:
    """Devuelve la ruta absoluta de una carpeta definida en cfg['rutas'] y la crea si no existe."""
    p = RAIZ / cfg["rutas"][clave]
    p.mkdir(parents=True, exist_ok=True)
    return p


def fijar_semillas(semilla: int) -> None:
    """Fija las semillas de random y numpy (torch se fija en la fase de modelos)."""
    random.seed(semilla)
    np.random.seed(semilla)

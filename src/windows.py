"""
Muestras, ventanas, partición temporal y preprocesado (Fase 2).

  * Una muestra se identifica por su día t (en el calendario del Brent). Sus entradas son las
    filas t-59..t de la matriz de variables y su objetivo es y(t) = [y_5, y_63, y_252].
  * Partición walk-forward expansiva con purga de 252 días hábiles: el entrenamiento termina
    252 días antes del inicio de validación/test, para que ninguna etiqueta de entrenamiento
    mire dentro del periodo evaluado.
  * El preprocesado (winsorización + StandardScaler de entradas y estandarización del objetivo
    por horizonte) se ajusta SOLO con el entrenamiento de cada partición.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------
# Muestras válidas
# ---------------------------------------------------------------
def posiciones_validas(X: pd.DataFrame, ventana: int) -> np.ndarray:
    """
    Posiciones t (enteras, en el calendario) cuya ventana t-ventana+1..t no tiene ningún NaN.
    No se exige que el objetivo exista: eso se decide después según el uso de la muestra.
    """
    completa = X.notna().all(axis=1).to_numpy().astype(int)
    # Nº de filas completas en la ventana que termina en t (suma móvil)
    suma = np.convolve(completa, np.ones(ventana, dtype=int), mode="full")[: len(completa)]
    return np.where(suma == ventana)[0]


# ---------------------------------------------------------------
# Partición temporal
# ---------------------------------------------------------------
@dataclass
class Particion:
    nombre: str                 # "pliegue_2018", ..., "final"
    pos_train: np.ndarray       # posiciones t de entrenamiento
    pos_eval: np.ndarray        # posiciones t de validación (pliegues) o test (final)
    fin_train: pd.Timestamp     # último día t de entrenamiento


def _corte_purgado(calendario: pd.DatetimeIndex, inicio: pd.Timestamp, purga: int) -> int:
    """Posición límite: el entrenamiento usa t < calendario[pos_inicio - purga]."""
    pos_inicio = int(calendario.searchsorted(inicio))
    return pos_inicio - purga


def crear_particiones(calendario: pd.DatetimeIndex, pos_validas: np.ndarray,
                      Y: pd.DataFrame, cfg: dict) -> list[Particion]:
    """Crea los pliegues de validación y la partición final (train final + test)."""
    cp = cfg["particion"]
    purga = cp["purga"]
    # Para entrenar y validar se exigen las 3 etiquetas (las tienen todas hasta 2022)
    con_objetivo = Y.notna().all(axis=1).to_numpy()
    algun_objetivo = Y.notna().any(axis=1).to_numpy()
    fechas = calendario[pos_validas]

    particiones = []
    for anio in cp["anios_validacion"]:
        ini, fin = pd.Timestamp(f"{anio}-01-01"), pd.Timestamp(f"{anio + 1}-01-01")
        limite = _corte_purgado(calendario, ini, purga)
        tr = pos_validas[(pos_validas < limite) & con_objetivo[pos_validas]]
        va = pos_validas[(fechas >= ini) & (fechas < fin) & con_objetivo[pos_validas]]
        particiones.append(Particion(f"pliegue_{anio}", tr, va, calendario[tr[-1]]))

    ini_test = pd.Timestamp(cp["inicio_test"])
    limite = _corte_purgado(calendario, ini_test, purga)
    tr = pos_validas[(pos_validas < limite) & con_objetivo[pos_validas]]
    # En test basta con conocer al menos un horizonte (la máscara ignora el resto)
    te = pos_validas[(fechas >= ini_test) & algun_objetivo[pos_validas]]
    particiones.append(Particion("final", tr, te, calendario[tr[-1]]))
    return particiones


def tabla_particiones(particiones: list[Particion], calendario: pd.DatetimeIndex,
                      Y: pd.DataFrame) -> pd.DataFrame:
    """Fechas exactas y nº de muestras de cada partición."""
    filas = []
    for p in particiones:
        fila = {
            "particion": p.nombre,
            "train_inicio": calendario[p.pos_train[0]].date(),
            "train_fin": calendario[p.pos_train[-1]].date(),
            "n_train": len(p.pos_train),
            "eval": "test" if p.nombre == "final" else "validacion",
            "eval_inicio": calendario[p.pos_eval[0]].date(),
            "eval_fin": calendario[p.pos_eval[-1]].date(),
            "n_eval": len(p.pos_eval),
            "dias_purga": int(p.pos_eval[0] - p.pos_train[-1] - 1),
        }
        for col in Y.columns:
            fila[f"n_eval_{col}"] = int(Y[col].iloc[p.pos_eval].notna().sum())
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------
# Preprocesado ajustado solo con entrenamiento
# ---------------------------------------------------------------
class Preprocesador:
    """Winsorización (percentiles de train) + StandardScaler de entradas; estandarización del objetivo."""

    def __init__(self, p_inf: float, p_sup: float, excluir: list[str]):
        self.p_inf, self.p_sup, self.excluir = p_inf, p_sup, set(excluir)

    def ajustar(self, X: pd.DataFrame, Y: pd.DataFrame, pos_train: np.ndarray, ventana: int):
        # Filas de entradas que ve el entrenamiento: desde el inicio de la primera ventana
        # hasta el último día t de entrenamiento
        filas = X.iloc[pos_train[0] - ventana + 1: pos_train[-1] + 1]
        self.columnas = list(X.columns)
        self.lim_inf = filas.quantile(self.p_inf)
        self.lim_sup = filas.quantile(self.p_sup)
        for c in self.excluir:  # sin winsorizar: límites infinitos
            self.lim_inf[c], self.lim_sup[c] = -np.inf, np.inf
        self.scaler = StandardScaler().fit(self._winsorizar(filas))
        y_tr = Y.iloc[pos_train]
        self.y_media = y_tr.mean().to_numpy()
        self.y_desv = y_tr.std(ddof=0).to_numpy()
        return self

    def _winsorizar(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columnas].clip(lower=self.lim_inf, upper=self.lim_sup, axis=1)

    def transformar_X(self, X: pd.DataFrame) -> np.ndarray:
        """Aplica winsorización + escalado a todas las filas (los NaN se mantienen)."""
        return self.scaler.transform(self._winsorizar(X)).astype(np.float32)

    def estandarizar_y(self, Y: np.ndarray) -> np.ndarray:
        return ((Y - self.y_media) / self.y_desv).astype(np.float32)

    def desestandarizar_y(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.y_desv + self.y_media

    def a_dict(self) -> dict:
        """Parámetros en formato JSON (para guardar junto al modelo)."""
        def lista(v):
            return [None if not np.isfinite(x) else float(x) for x in v]
        return {
            "columnas": self.columnas,
            "winsor_inf": lista(self.lim_inf[self.columnas].to_numpy()),
            "winsor_sup": lista(self.lim_sup[self.columnas].to_numpy()),
            "scaler_media": self.scaler.mean_.tolist(),
            "scaler_escala": self.scaler.scale_.tolist(),
            "y_media": self.y_media.tolist(),
            "y_desv": self.y_desv.tolist(),
            "p_inf": self.p_inf, "p_sup": self.p_sup, "excluir": sorted(self.excluir),
        }

    @classmethod
    def desde_dict(cls, d: dict) -> "Preprocesador":
        """Reconstruye el preprocesador a partir de a_dict() (para la app)."""
        p = cls(d["p_inf"], d["p_sup"], d["excluir"])
        p.columnas = d["columnas"]
        p.lim_inf = pd.Series([-np.inf if v is None else v for v in d["winsor_inf"]], index=p.columnas)
        p.lim_sup = pd.Series([np.inf if v is None else v for v in d["winsor_sup"]], index=p.columnas)
        p.scaler = StandardScaler()
        p.scaler.mean_ = np.array(d["scaler_media"])
        p.scaler.scale_ = np.array(d["scaler_escala"])
        p.scaler.var_ = p.scaler.scale_ ** 2
        p.scaler.n_features_in_ = len(p.columnas)
        p.scaler.feature_names_in_ = np.array(p.columnas, dtype=object)
        p.y_media, p.y_desv = np.array(d["y_media"]), np.array(d["y_desv"])
        return p


def construir_ventanas(Xn: np.ndarray, posiciones: np.ndarray, ventana: int) -> np.ndarray:
    """Tensor (n_muestras, ventana, n_variables) con las filas t-ventana+1..t de cada muestra."""
    desplaz = np.arange(-ventana + 1, 1)
    return Xn[posiciones[:, None] + desplaz[None, :]]

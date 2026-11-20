"""
Redes recurrentes y bucle de entrenamiento (PyTorch, CPU).

Las tres arquitecturas son idénticas salvo la celda:
    capa recurrente (RNN tanh / LSTM / GRU, 1 capa, batch_first) -> Dropout -> Linear(units, 3)
Se usa el estado oculto del último paso de la ventana (día t).
"""
import copy
import time

import numpy as np
import torch
from torch import nn

CELDAS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


def fijar_semillas_torch(semilla: int) -> None:
    """Semillas de numpy y torch + algoritmos deterministas en CPU."""
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(True)


class RedRecurrente(nn.Module):
    def __init__(self, tipo: str, n_entradas: int, units: int, dropout: float, n_salidas: int = 3):
        super().__init__()
        extra = {"nonlinearity": "tanh"} if tipo == "rnn" else {}
        self.tipo = tipo
        self.recurrente = CELDAS[tipo](n_entradas, units, num_layers=1, batch_first=True, **extra)
        self.dropout = nn.Dropout(dropout)
        self.salida = nn.Linear(units, n_salidas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        secuencia, _ = self.recurrente(x)          # (batch, ventana, units)
        ultimo = secuencia[:, -1, :]               # estado en el día t
        return self.salida(self.dropout(ultimo))   # (batch, 3)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# ---------------------------------------------------------------
# Pérdidas con máscara (los NaN del objetivo se ignoran)
# ---------------------------------------------------------------
def mse_mascara(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE medio sobre todos los elementos conocidos del objetivo."""
    mascara = ~torch.isnan(y)
    dif = torch.where(mascara, pred - torch.nan_to_num(y), torch.zeros_like(pred))
    return (dif ** 2).sum() / mascara.sum().clamp(min=1)


def mse_por_horizonte(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MSE de cada horizonte (columna) ignorando NaN."""
    return np.nanmean((pred - y) ** 2, axis=0)


# ---------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------
@torch.no_grad()
def predecir(modelo: nn.Module, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Predicciones (estandarizadas) en modo evaluación."""
    modelo.eval()
    salidas = [modelo(torch.from_numpy(X[i:i + batch])) for i in range(0, len(X), batch)]
    return torch.cat(salidas).numpy()


def entrenar(modelo: nn.Module, X_tr: np.ndarray, Y_tr: np.ndarray, cfg_ent: dict, semilla: int,
             X_va: np.ndarray | None = None, Y_va: np.ndarray | None = None,
             epocas: int | None = None, log=None) -> dict:
    """
    Entrena con Adam, mini-lotes barajados (solo dentro del entrenamiento) y gradient clipping.

    * Con validación: hasta `max_epocas` con early stopping (paciencia) sobre la pérdida de
      validación = media de los MSE por horizonte; se restauran los mejores pesos.
    * Sin validación: exactamente `epocas` épocas (entrenamiento final).
    Devuelve el historial y la época óptima (base 1).
    """
    fijar_semillas_torch(semilla)
    gen = torch.Generator().manual_seed(semilla)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg_ent["lr"])
    Xt, Yt = torch.from_numpy(X_tr), torch.from_numpy(Y_tr)
    n, bs = len(Xt), cfg_ent["batch"]
    con_val = X_va is not None
    n_epocas = cfg_ent["max_epocas"] if con_val else epocas

    hist = {"train": [], "val": []}
    mejor, mejor_epoca, mejores_pesos, sin_mejora = np.inf, 0, None, 0
    t0 = time.perf_counter()
    for epoca in range(1, n_epocas + 1):
        modelo.train()
        orden = torch.randperm(n, generator=gen)
        suma, cuenta = 0.0, 0
        for i in range(0, n, bs):
            idx = orden[i:i + bs]
            opt.zero_grad()
            perdida = mse_mascara(modelo(Xt[idx]), Yt[idx])
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg_ent["clip"])
            opt.step()
            suma += perdida.item() * len(idx)
            cuenta += len(idx)
        hist["train"].append(suma / cuenta)   # pérdida media de la época (con dropout activo)

        if con_val:
            val = float(mse_por_horizonte(predecir(modelo, X_va), Y_va).mean())
            hist["val"].append(val)
            if val < mejor - 1e-7:
                mejor, mejor_epoca, sin_mejora = val, epoca, 0
                mejores_pesos = copy.deepcopy(modelo.state_dict())
            else:
                sin_mejora += 1
            if sin_mejora >= cfg_ent["paciencia"]:
                break
    if con_val:
        modelo.load_state_dict(mejores_pesos)
    else:
        mejor_epoca = n_epocas

    return {"historial": hist, "epoca_optima": mejor_epoca, "perdida_val": mejor if con_val else None,
            "epocas_ejecutadas": len(hist["train"]), "segundos": time.perf_counter() - t0}

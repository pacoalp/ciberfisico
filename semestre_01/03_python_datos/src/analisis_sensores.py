"""Simulación y análisis de lecturas IoT de calidad de agua.

Genera un lote de observaciones de oxígeno disuelto (mg/L) y temperatura (°C)
y reporta la media muestral de cada variable.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

N_LECTURAS: Final[int] = 10
SEMILLA: Final[int] = 42

# Rangos físicos típicos de un sondeo superficial (agua dulce, clima templado).
DO_MG_L_MIN: Final[float] = 4.0
DO_MG_L_MAX: Final[float] = 12.0
TEMP_C_MIN: Final[float] = 18.0
TEMP_C_MAX: Final[float] = 28.0

COLUMNAS: Final[tuple[str, str, str]] = (
    "timestamp",
    "oxigeno_disuelto_mg_l",
    "temperatura_c",
)


def generar_lecturas(
    n: int = N_LECTURAS,
    semilla: int = SEMILLA,
) -> pd.DataFrame:
    """Construye un DataFrame con n lecturas simuladas del sensor.

    El muestreo es uniforme e independiente en los intervalos físicos
    [DO_MG_L_MIN, DO_MG_L_MAX] y [TEMP_C_MIN, TEMP_C_MAX]. La semilla
    fija el generador para que el lote sea reproducible.
    """
    if n <= 0:
        raise ValueError(f"n debe ser un entero positivo; se recibió {n}.")

    rng = np.random.default_rng(semilla)
    timestamps = pd.date_range(
        start="2026-10-28 08:00:00",
        periods=n,
        freq="15min",
        tz="America/Mexico_City",
    )
    oxigeno = rng.uniform(DO_MG_L_MIN, DO_MG_L_MAX, size=n)
    temperatura = rng.uniform(TEMP_C_MIN, TEMP_C_MAX, size=n)

    return pd.DataFrame(
        {
            COLUMNAS[0]: timestamps,
            COLUMNAS[1]: np.round(oxigeno, 2),
            COLUMNAS[2]: np.round(temperatura, 2),
        }
    )


def media_variables(df: pd.DataFrame) -> pd.Series:
    """Media aritmética muestral: (1/n) * sum(x_i) por variable numérica.

    Excluye timestamp. Si alguna columna numérica está vacía o es todo NaN,
    Pandas devolverá NaN en esa media (skipna=True por defecto).
    """
    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        raise ValueError("El DataFrame no contiene columnas numéricas.")
    return numericas.mean()


def main() -> None:
    lecturas = generar_lecturas()
    medias = media_variables(lecturas)

    print("Lecturas simuladas del sensor de calidad de agua")
    print(lecturas.to_string(index=False))
    print()
    print("Medias muestrales")
    print(
        f"  oxígeno disuelto: {medias[COLUMNAS[1]]:.2f} mg/L"
    )
    print(f"  temperatura:      {medias[COLUMNAS[2]]:.2f} °C")


if __name__ == "__main__":
    main()

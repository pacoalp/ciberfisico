"""Simulación y análisis de lecturas IoT de calidad de agua.

Genera un lote de observaciones de oxígeno disuelto (mg/L) y temperatura (°C)
y reporta media y desviación estándar muestral de cada variable.
"""

from __future__ import annotations

from typing import Final, TypedDict

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


class EstadisticosVariable(TypedDict):
    """Parámetros muestrales de una variable numérica."""

    media: float
    desviacion_estandar: float


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


def media_variables(df: pd.DataFrame) -> dict[str, EstadisticosVariable]:
    """Media y desviación estándar muestral por variable numérica.

    Media: (1/n) * sum(x_i).
    Desviación estándar muestral (corrección de Bessel, ddof=1):
        s = sqrt( 1/(n-1) * sum((x_i - media)^2) ).

    Excluye timestamp. Con skipna=True, NaN se omiten. Si n < 2 o la
    columna es todo NaN, la desviación estándar queda como NaN.
    """
    numericas = df.select_dtypes(include="number")
    if numericas.empty:
        raise ValueError("El DataFrame no contiene columnas numéricas.")

    medias = numericas.mean()
    desviaciones = numericas.std(ddof=1)
    return {
        columna: EstadisticosVariable(
            media=float(medias[columna]),
            desviacion_estandar=float(desviaciones[columna]),
        )
        for columna in numericas.columns
    }


def main() -> None:
    lecturas = generar_lecturas()
    estadisticos = media_variables(lecturas)
    oxigeno = estadisticos[COLUMNAS[1]]
    temperatura = estadisticos[COLUMNAS[2]]

    print("Lecturas simuladas del sensor de calidad de agua")
    print(lecturas.to_string(index=False))
    print()
    print("Estadísticos muestrales")
    print(
        "  oxígeno disuelto: "
        f"media {oxigeno['media']:.2f} mg/L, "
        f"s {oxigeno['desviacion_estandar']:.2f} mg/L"
    )
    print(
        "  temperatura:      "
        f"media {temperatura['media']:.2f} °C, "
        f"s {temperatura['desviacion_estandar']:.2f} °C"
    )


if __name__ == "__main__":
    main()

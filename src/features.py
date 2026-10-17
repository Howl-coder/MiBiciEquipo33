from pathlib import Path
import numpy as np
import pandas as pd
from config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

def generar_malla_demanda_horaria(df_viajes: pd.DataFrame, ids_estaciones: list) -> pd.DataFrame:
    """Construye la malla regular hora x estación con imputación de demanda cero."""
    # Truncar inicio a la hora
    df_viajes["timestamp_hora"] = df_viajes["Inicio_del_viaje"].dt.floor("h")

    # Demanda de salidas por estación y hora
    demanda = (
        df_viajes.groupby(["timestamp_hora", "Origen_Id"])
        .size()
        .reset_index(name="demanda_salidas")
    )

    # Flujo de llegadas por estación y hora (para P2)
    df_viajes["timestamp_fin_hora"] = df_viajes["Fin_del_viaje"].dt.floor("h")
    llegadas = (
        df_viajes.groupby(["timestamp_fin_hora", "Destino_Id"])
        .size()
        .reset_index(name="flujo_llegadas")
        .rename(columns={"timestamp_fin_hora": "timestamp_hora", "Destino_Id": "Origen_Id"})
    )

    # Rango horario completo continuo
    rango_horas = pd.date_range(
        start=df_viajes["timestamp_hora"].min(),
        end=df_viajes["timestamp_hora"].max(),
        freq="h"
    )

    # Producto cartesiano completo
    indice_cartesiano = pd.MultiIndex.from_product(
        [rango_horas, ids_estaciones], names=["timestamp_hora", "Origen_Id"]
    ).to_frame().reset_index(drop=True)

    # Combinar y rellenar ceros
    malla = indice_cartesiano.merge(demanda, on=["timestamp_hora", "Origen_Id"], how="left")
    malla = malla.merge(llegadas, on=["timestamp_hora", "Origen_Id"], how="left")

    malla["demanda_salidas"] = malla["demanda_salidas"].fillna(0).astype(int)
    malla["flujo_llegadas"] = malla["flujo_llegadas"].fillna(0).astype(int)

    malla.sort_values(by=["Origen_Id", "timestamp_hora"], inplace=True)
    malla.reset_index(drop=True, inplace=True)
    return malla

def calcular_features_temporales_y_lags(df_malla: pd.DataFrame) -> pd.DataFrame:
    """Calcula transformaciones cíclicas, variables de calendario y retardos."""
    df = df_malla.copy()

    # Features cíclicas
    hora = df["timestamp_hora"].dt.hour
    dia_semana = df["timestamp_hora"].dt.dayofweek

    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    df["dia_sin"] = np.sin(2 * np.pi * dia_semana / 7)
    df["dia_cos"] = np.cos(2 * np.pi * dia_semana / 7)
    df["es_fin_de_semana"] = (dia_semana >= 5).astype(int)

    # Rezagos temporales por estación (1h, 2h, 3h, 24h, 168h)
    for lag in [1, 2, 3, 24, 168]:
        df[f"lag_salidas_{lag}h"] = df.groupby("Origen_Id")["demanda_salidas"].shift(lag)

    # Media móvil de las últimas 6 horas (shift(1) para evitar data leakage)
    df["rolling_mean_6h"] = df.groupby("Origen_Id")["demanda_salidas"].transform(
        lambda x: x.shift(1).rolling(6).mean()
    )

    # Balance instantáneo para P2
    df["balance_neto_h"] = df["flujo_llegadas"] - df["demanda_salidas"]

    # Descartar filas con NaN derivados de los lags iniciales
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    ruta_interim = INTERIM_DATA_DIR / "viajes_amg_limpios.parquet"
    if not ruta_interim.exists():
        raise FileNotFoundError(f"Primero corre dataset.py para generar {ruta_interim}")

    print(f"Cargando {ruta_interim.name}...")
    df_viajes = pd.read_parquet(ruta_interim)

    estaciones = sorted(list(df_viajes["Origen_Id"].unique()))
    print(f"Generando malla horaria para {len(estaciones)} estaciones...")
    df_malla = generar_malla_demanda_horaria(df_viajes, estaciones)

    print("Calculando features temporales, cíclicas y lags...")
    df_features = calcular_features_temporales_y_lags(df_malla)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    salida = PROCESSED_DATA_DIR / "features_demanda_p1.parquet"
    df_features.to_parquet(salida, index=False)

    print(f"Features P1 guardadas con éxito en: {salida}")
    print(f"Dimensiones de entrenamiento: {df_features.shape}")
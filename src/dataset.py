import gc
from pathlib import Path
import pandas as pd
from config import (
    ARCHIVOS_VIAJES_RAW,
    RUTA_CATALOGO_ESTACIONES,
    INTERIM_DATA_DIR,
    FECHA_INICIO,
    FECHA_FIN,
    DURACION_MIN_MINUTOS,
    DURACION_MAX_MINUTOS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
)

def cargar_catalogo_amg(ruta_estaciones: Path) -> pd.DataFrame:
    """Carga y filtra estaciones estrictamente dentro del polígono AMG."""
    try:
        df_est = pd.read_csv(ruta_estaciones, encoding="utf-8")
    except UnicodeDecodeError:
        df_est = pd.read_csv(ruta_estaciones, encoding="latin-1")
    
    # Filtrado por coordenadas AMG
    mask_amg = (
        (df_est["latitude"] >= LAT_MIN) & (df_est["latitude"] <= LAT_MAX) &
        (df_est["longitude"] >= LON_MIN) & (df_est["longitude"] <= LON_MAX)
    )
    df_est = df_est[mask_amg].copy()
    df_est["id"] = df_est["id"].astype(int)
    return df_est

def procesar_e_integrar_datos(archivos: list, ruta_estaciones: Path) -> pd.DataFrame:
    """Lee archivos explícitos, limpia nombres, optimiza memoria y filtra al AMG."""
    df_est = cargar_catalogo_amg(ruta_estaciones)
    estaciones_validas = set(df_est["id"].unique())
    print(f"Estaciones válidas en AMG: {len(estaciones_validas)}")

    dfs = []
    for f in archivos:
        print(f"Leyendo: {f.name}...")
        try:
            temp = pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            temp = pd.read_csv(f, encoding="latin-1")

        # Unificar nombres de columnas
        temp.columns = (
            temp.columns.str.replace("Año", "Anio")
            .str.replace("AÃ±o", "Anio")
            .str.strip()
        )

        if "Genero" in temp.columns:
            temp["Genero"] = temp["Genero"].astype("category")

        dfs.append(temp)

    # Concatenar y liberar memoria
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    print(f"Total registros cargados sin deduplicar: {df.shape[0]:,}")

    # Deduplicación por Viaje_Id
    df.drop_duplicates(subset=["Viaje_Id"], keep="first", inplace=True)

    # Conversión eficiente de fechas
    df["Inicio_del_viaje"] = pd.to_datetime(
        df["Inicio_del_viaje"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["Fin_del_viaje"] = pd.to_datetime(
        df["Fin_del_viaje"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    # Filtro temporal
    df = df[
        (df["Inicio_del_viaje"] >= FECHA_INICIO) & 
        (df["Inicio_del_viaje"] <= FECHA_FIN)
    ]

    # Duración en minutos y filtros de calidad (1 min a 24 horas)
    duracion = (df["Fin_del_viaje"] - df["Inicio_del_viaje"]).dt.total_seconds() / 60.0
    df["duracion_minutos"] = duracion
    df = df[
        (df["duracion_minutos"] >= DURACION_MIN_MINUTOS) & 
        (df["duracion_minutos"] <= DURACION_MAX_MINUTOS)
    ]

    # Filtro de red: tanto origen como destino deben estar en estaciones válidas de AMG
    df = df[
        df["Origen_Id"].isin(estaciones_validas) & 
        df["Destino_Id"].isin(estaciones_validas)
    ]

    # Ordenar y resetear índice in-place
    df.sort_values(by="Inicio_del_viaje", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Guardar en data/interim
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    salida = INTERIM_DATA_DIR / "viajes_amg_limpios.parquet"
    df.to_parquet(salida, index=False)
    
    print(f"Guardado exitoso en {salida}")
    print(f"Dimensiones finales: {df.shape}")
    print(f"Uso de memoria: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    return df

if __name__ == "__main__":
    procesar_e_integrar_datos(ARCHIVOS_VIAJES_RAW, RUTA_CATALOGO_ESTACIONES)
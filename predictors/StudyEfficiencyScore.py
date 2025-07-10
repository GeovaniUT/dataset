from utils.data_helpers.StudyEfficiencyDataHelper import GenerarDatosStudyEfficiency
from utils.plot_utils import PlotToBase64
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from model_utils_excel import cargar_pkl_enriquecido
from pathlib import Path
import joblib

# Ruta modelo persistente
_MODELO_DIR = Path(__file__).resolve().parent.parent / "modelos" / "study_efficiency"
_MODELO_DIR.mkdir(parents=True, exist_ok=True)
_MODELO_PATH = _MODELO_DIR / "gbr.pkl"


def CompletarStudyEfficiencyScore(df):
    sintetico = GenerarDatosStudyEfficiency(6000)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(
        sintetico[["AffectsAcademicPerformance", "AvgDailyUsageHours", "SleepHoursPerNight"]],
        sintetico["StudyEfficiencyScore"],
    )

    mask = df["StudyEfficiencyScore"].isna()
    if mask.any():
        X_pred = df.loc[mask, [
            "AffectsAcademicPerformance",
            "AvgDailyUsageHours",
            "SleepHoursPerNight",
        ]]
        df.loc[mask, "StudyEfficiencyScore"] = model.predict(X_pred).round().clip(0, 100)

    return df


def PredecirStudyEfficiencyScore(affects_academic: int, avg_usage: float, sleep_hours: float):
    # --- Cargar datos reales ---
    df = cargar_pkl_enriquecido()

    # Normalizar y mapear columnas a PascalCase
    df.columns = df.columns.str.strip()  # quitar espacios accidentales
    column_map = {
        "affects_academic_performance": "AffectsAcademicPerformance",
        "affects _academic_performance": "AffectsAcademicPerformance",
        "avg_daily_usage_hours": "AvgDailyUsageHours",
        "sleep_hours_per_night": "SleepHoursPerNight",
    }

    for original, target in column_map.items():
        if target not in df.columns and original in df.columns:
            df[target] = df[original]

    if "StudyEfficiencyScore" not in df.columns:
        # Derivar con fórmula heurística de helper
        df_generated = GenerarDatosStudyEfficiency(len(df))
        df["StudyEfficiencyScore"] = df_generated["StudyEfficiencyScore"]

    # --- Modelo ---
    if _MODELO_PATH.exists():
        model = joblib.load(_MODELO_PATH)
    else:
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(
            df[["AffectsAcademicPerformance", "AvgDailyUsageHours", "SleepHoursPerNight"]],
            df["StudyEfficiencyScore"],
        )
        joblib.dump(model, _MODELO_PATH)

    pred_score = int(model.predict([[affects_academic, avg_usage, sleep_hours]])[0].round().clip(0, 100))

    # --- Gráfica: barra horizontal segmentada ---
    fig, ax = plt.subplots(figsize=(7, 2.5))
    # Segmentos de color
    zones = [(0, 60, "#e74c3c"), (60, 80, "#f1c40f"), (80, 100, "#2ecc71")]
    for start, end, color in zones:
        ax.barh(0, end - start, left=start, color=color)

    # Línea del usuario
    ax.axvline(pred_score, color="black", linewidth=2)
    ax.text(pred_score, 0.1, f"{pred_score}%", ha="center", va="bottom", fontsize=10)

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Eficiencia (%)")
    ax.set_title("Eficiencia de Estudio")
    ax.grid(axis="x", alpha=0.2, linestyle=":")

    graph = PlotToBase64(fig)

    return {
        "StudyEfficiencyScore": pred_score,
        "Grafica": f"data:image/png;base64,{graph}",
        "Entrada": {
            "AffectsAcademicPerformance": affects_academic,
            "AvgDailyUsageHours": avg_usage,
            "SleepHoursPerNight": sleep_hours,
        },
        "modelo_metadata": {
            "algoritmo": "RandomForest",
            "variables": [
                "affects_academic_performance",
                "avg_daily_usage_hours",
                "sleep_hours_per_night",
            ],
            "explicacion_modelo": {
                "que_es": "Bosque aleatorio que estima tu eficiencia de estudio en porcentaje.",
                "como_funciona": "Promedia muchos árboles de decisión para mejorar la precisión y evitar sobreajuste.",
                "para_que_sirve": "Mostrar cómo tus hábitos influyen en tu rendimiento académico y sugerir mejoras."
            },
            "modelo_persistente": True,
        },
        "interpretacion_graficas": {
            "barra_segmentada": "La barra muestra tres zonas: rojo (ineficiente), amarillo (medio) y verde (eficiente). La línea negra indica tu porcentaje actual."
        }
    } 
from utils.data_helpers.ConflictRiskDataHelper import GenerarDatosConflictRisk
from utils.plot_utils import PlotToBase64
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from model_utils_excel import cargar_pkl_enriquecido
from pathlib import Path
import joblib


# Ruta para el modelo persistente
_MODELO_DIR = Path(__file__).resolve().parent.parent / "modelos" / "conflict_risk"
_MODELO_DIR.mkdir(parents=True, exist_ok=True)
_MODELO_PATH = _MODELO_DIR / "random_forest.pkl"


def CompletarConflictRisk(df):
    """Completa la columna ConflictRisk con 0/1."""
    sintetico = GenerarDatosConflictRisk(6000)
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(
        sintetico[["AddictedScore", "RelationshipStatus", "AvgDailyUsageHours", "MostUsedPlatform"]],
        sintetico["ConflictRisk"],
    )

    mask = df["ConflictRisk"].isna()
    if mask.any():
        X_pred = df.loc[mask, [
            "AddictedScore",
            "RelationshipStatus",
            "AvgDailyUsageHours",
            "MostUsedPlatform",
        ]]
        df.loc[mask, "ConflictRisk"] = model.predict(X_pred)

    return df


def PredecirConflictRisk(addicted_score: int, avg_usage: float, relationship_status: int, platform: int):
    """Predice riesgo de conflicto usando datos reales y modelo persistente."""

    # Cargar datos reales
    ComunidadDF = cargar_pkl_enriquecido()

    # Mapear nombres a PascalCase
    if "AddictedScore" not in ComunidadDF.columns and "addicted_score" in ComunidadDF.columns:
        ComunidadDF["AddictedScore"] = ComunidadDF["addicted_score"]
    if "AvgDailyUsageHours" not in ComunidadDF.columns and "avg_daily_usage_hours" in ComunidadDF.columns:
        ComunidadDF["AvgDailyUsageHours"] = ComunidadDF["avg_daily_usage_hours"]
    if "RelationshipStatus" not in ComunidadDF.columns and "relationship_status" in ComunidadDF.columns:
        ComunidadDF["RelationshipStatus"] = ComunidadDF["relationship_status"]
    if "MostUsedPlatform" not in ComunidadDF.columns and "most_used_platform" in ComunidadDF.columns:
        ComunidadDF["MostUsedPlatform"] = ComunidadDF["most_used_platform"]

    # Convertir plataformas texto→código
    plataforma_map = {
        "Facebook": 1,
        "Instagram": 2,
        "Twitter": 3,
        "TikTok": 4,
        "YouTube": 5,
        "LinkedIn": 6,
        "Snapchat": 7,
        "WhatsApp": 8,
        "Otra": 9,
    }
    ComunidadDF["MostUsedPlatform"] = ComunidadDF["MostUsedPlatform"].map(lambda x: plataforma_map.get(x, 9))

    # Etiqueta objetivo
    if "ConflictRisk" not in ComunidadDF.columns:
        if "conflicts_over_social_media" in ComunidadDF.columns:
            ComunidadDF["ConflictRisk"] = ComunidadDF["conflicts_over_social_media"]
        else:
            # Si no existe, derivar heurísticamente usando helper
            SinteticoTemp = GenerarDatosConflictRisk(len(ComunidadDF))
            ComunidadDF["ConflictRisk"] = SinteticoTemp["ConflictRisk"]

    # Cargar o entrenar modelo
    if _MODELO_PATH.exists():
        model = joblib.load(_MODELO_PATH)
    else:
        model = RandomForestClassifier(n_estimators=120, random_state=42)
        model.fit(
            ComunidadDF[[
                "AddictedScore",
                "RelationshipStatus",
                "AvgDailyUsageHours",
                "MostUsedPlatform",
            ]],
            ComunidadDF["ConflictRisk"],
        )
        joblib.dump(model, _MODELO_PATH)

    prob = model.predict_proba([[addicted_score, relationship_status, avg_usage, platform]])[0][1]
    pred = int(prob >= 0.5)

    # --- Gráfica: CDF de probabilidad de conflicto ---
    comunidad_probs = model.predict_proba(
        ComunidadDF[[
            "AddictedScore",
            "RelationshipStatus",
            "AvgDailyUsageHours",
            "MostUsedPlatform",
        ]]
    )[:, 1]

    cdf_x = np.sort(comunidad_probs)
    cdf_y = np.linspace(0, 1, len(cdf_x))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cdf_x, cdf_y, label="CDF Comunidad", color="#3498db")

    # Línea vertical del usuario
    ax.axvline(prob, color="#e74c3c", linestyle="--", linewidth=2, label=f"Tú: {prob*100:.1f}%")

    # Línea horizontal para mostrar percentil
    percentile = (cdf_x < prob).mean()
    ax.axhline(percentile, color="#e74c3c", linestyle=":", linewidth=1)
    ax.scatter([prob], [percentile], color="#e74c3c")
    ax.annotate(
        f"{percentile*100:.0f}º percentil",
        (prob, percentile),
        textcoords="offset points",
        xytext=(10, -10),
        ha="left",
        color="#e74c3c",
    )

    ax.set_xlabel("Probabilidad de Conflicto")
    ax.set_ylabel("Proporción acumulada")
    ax.set_title("Distribución de Probabilidad de Conflictos (CDF)")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend()

    graph = PlotToBase64(fig)

    return {
        "RiesgoAlto": bool(pred),
        "Probabilidad": prob,
        "Grafica": f"data:image/png;base64,{graph}",
        "Entrada": {
            "AddictedScore": addicted_score,
            "AvgDailyUsageHours": avg_usage,
            "RelationshipStatus": relationship_status,
            "MostUsedPlatform": platform,
        },
        "modelo_metadata": {
            "algoritmo": "RandomForestClassifier",
            "variables": [
                "addicted_score",
                "relationship_status",
                "avg_daily_usage_hours",
                "most_used_platform",
            ],
            "explicacion_modelo": {
                "que_es": "Clasificador de bosque aleatorio que evalúa factores de uso y relación para estimar riesgo de conflictos.",
                "como_funciona": "Crea múltiples árboles de decisión y combina sus resultados para una predicción robusta.",
                "para_que_sirve": "Determinar la probabilidad de que tu actividad en redes genere conflictos interpersonales."
            },
            "modelo_persistente": True
        },
        "interpretacion_graficas": {
            "cdf": "La curva azul muestra la distribución de riesgo en la comunidad. La línea roja marca tu probabilidad: mientras más a la derecha, mayor riesgo comparado con el resto."
        }
    } 
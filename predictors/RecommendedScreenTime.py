from utils.data_helpers.RecommendedScreenTimeDataHelper import GenerarDatosRecommendedScreenTime
from utils.plot_utils import PlotToBase64
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from model_utils_excel import cargar_pkl_enriquecido
from pathlib import Path
import joblib

# Ruta para guardar/cargar el modelo entrenado
_MODELO_DIR = Path(__file__).resolve().parent.parent / "modelos" / "recommended_screen_time"
_MODELO_DIR.mkdir(parents=True, exist_ok=True)
_MODELO_PATH = _MODELO_DIR / "linear_regression.pkl"


def CompletarRecommendedScreenTime(df):
    """Completa columna RecommendedScreenTime (horas)."""
    sintetico = GenerarDatosRecommendedScreenTime(6000)
    model = LinearRegression()
    model.fit(
        sintetico[["Age", "AddictedScore", "MentalHealthScore"]],
        sintetico["RecommendedScreenTime"],
    )

    mask = df["RecommendedScreenTime"].isna()
    if mask.any():
        X_pred = df.loc[mask, ["Age", "AddictedScore", "MentalHealthScore"]]
        df.loc[mask, "RecommendedScreenTime"] = model.predict(X_pred).round(1)

    return df


def PredecirRecommendedScreenTime(Age: int, AddictedScore: int, MentalHealth: int):
    """Predice horas recomendadas de uso y genera una gráfica explicativa."""
    # Cargar o entrenar el modelo
    if _MODELO_PATH.exists():
        Modelo = joblib.load(_MODELO_PATH)
        SinteticoDF = cargar_pkl_enriquecido()  # Para gráfica comunitaria
    else:
        SinteticoDF = cargar_pkl_enriquecido()

    # Mapear nombres de columnas
    if "Age" not in SinteticoDF.columns and "age" in SinteticoDF.columns:
        SinteticoDF["Age"] = SinteticoDF["age"]
    if "AddictedScore" not in SinteticoDF.columns and "addicted_score" in SinteticoDF.columns:
        SinteticoDF["AddictedScore"] = SinteticoDF["addicted_score"]
    if "MentalHealthScore" not in SinteticoDF.columns and "mental_health_score" in SinteticoDF.columns:
        SinteticoDF["MentalHealthScore"] = SinteticoDF["mental_health_score"]
    if "RecommendedScreenTime" not in SinteticoDF.columns:
        if "recommended_screen_time" in SinteticoDF.columns:
            SinteticoDF["RecommendedScreenTime"] = SinteticoDF["recommended_screen_time"]
        else:
            # Calcular horas recomendadas con la fórmula heurística usada en los datos sintéticos
            SinteticoDF["RecommendedScreenTime"] = (
                2
                + 3 * (SinteticoDF["AddictedScore"] / 10)
                - 1.5 * (SinteticoDF["MentalHealthScore"] / 10)
                + SinteticoDF["Age"] / 60
            ).clip(0.5, 6).round(1)

    if not _MODELO_PATH.exists():
        Modelo = LinearRegression()
        Modelo.fit(
            SinteticoDF[["Age", "AddictedScore", "MentalHealthScore"]],
            SinteticoDF["RecommendedScreenTime"],
        )
        joblib.dump(Modelo, _MODELO_PATH)

    HorasPred = float(
        Modelo.predict([[Age, AddictedScore, MentalHealth]])[0].round(1)
    )

    # Crear gráfica
    Fig, Ax = plt.subplots(figsize=(8, 6))

    # Dispersión de la comunidad
    Ax.scatter(
        SinteticoDF["Age"],
        SinteticoDF["RecommendedScreenTime"],
        alpha=0.25,
        label="Comunidad",
        color="#3498db",
        edgecolors="none",
    )

    # Punto del usuario
    Ax.scatter(
        [Age],
        [HorasPred],
        color="red",
        label="Tú",
        s=120,
        marker="*",
        zorder=5,
    )

    # Línea de tendencia/óptima estimada por el modelo
    AgesLine = np.linspace(SinteticoDF["Age"].min(), SinteticoDF["Age"].max(), 100)
    PredsLine = Modelo.predict(
        np.column_stack(
            (
                AgesLine,
                np.full(100, AddictedScore),
                np.full(100, MentalHealth),
            )
        )
    )
    Ax.plot(
        AgesLine,
        PredsLine,
        color="black",
        linestyle="--",
        label="Línea óptima",
    )

    # Etiquetas y estilo
    Ax.set_xlabel("Edad")
    Ax.set_ylabel("Horas recomendadas")
    Ax.set_title("Uso de Pantalla Recomendado: Comunidad vs. Tú")
    Ax.grid(alpha=0.3, linestyle=":")
    Ax.legend()

    # Anotar el punto del usuario con el valor predicho
    Ax.annotate(
        f"{HorasPred} h",
        (Age, HorasPred),
        textcoords="offset points",
        xytext=(0, -15),
        ha="center",
        color="red",
        fontsize=10,
        weight="bold",
    )

    GraphBase64 = PlotToBase64(Fig)

    return {
        "RecommendedHours": HorasPred,
        "Grafica": f"data:image/png;base64,{GraphBase64}",
        "Entrada": {
            "Age": Age,
            "AddictedScore": AddictedScore,
            "MentalHealthScore": MentalHealth,
        },
        "ModeloMetadata": {
            "Algoritmo": "LinearRegression",
            "Variables": ["Age", "AddictedScore", "MentalHealthScore"],
            "R2": float(
                Modelo.score(
                    SinteticoDF[["Age", "AddictedScore", "MentalHealthScore"]],
                    SinteticoDF["RecommendedScreenTime"],
                )
            ),
        },
    } 
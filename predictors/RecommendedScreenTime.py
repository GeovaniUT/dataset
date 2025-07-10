from utils.plot_utils import PlotToBase64
import matplotlib.pyplot as plt
import numpy as np
from model_utils_excel import cargar_pkl_enriquecido
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor


# Ruta para guardar/cargar el modelo entrenado
_MODELO_DIR = Path(__file__).resolve().parent.parent / "modelos" / "recommended_screen_time"
_MODELO_DIR.mkdir(parents=True, exist_ok=True)
_MODELO_PATH = _MODELO_DIR / "random_forest_regressor.pkl"


def _get_or_train_model():
    """
    Carga el modelo de recomendación de tiempo en pantalla si existe.
    Si no, lo entrena usando una heurística mejorada y lo guarda.
    Retorna el modelo y los datos usados para la visualización.
    """
    if _MODELO_PATH.exists():
        modelo = joblib.load(_MODELO_PATH)
        datos = cargar_pkl_enriquecido() # Para la gráfica
        return modelo, datos

    # --- Si el modelo no existe, se entrena ---
    datos = cargar_pkl_enriquecido()

    # Mapeo de columnas para asegurar compatibilidad
    for col_lower in ["age", "addicted_score", "mental_health_score"]:
        col_pascal = col_lower.title().replace("_", "")
        if col_pascal not in datos.columns and col_lower in datos.columns:
            datos[col_pascal] = datos[col_lower]

    # --- Nueva Heurística para RecommendedScreenTime ---
    # 1. Baseline por edad
    def get_baseline(age):
        if age <= 18: return 2.0  # Más estricto para adolescentes
        elif 18 < age <= 25: return 3.0  # Adultos jóvenes
        else: return 2.5 # Adultos

    datos['BaselineHours'] = datos['Age'].apply(get_baseline)

    # 2. Ajuste por adicción (a mayor adicción, menor tiempo)
    # Reduce hasta 2.5 horas para el puntaje más alto
    addiction_adjustment = -2.5 * (datos['AddictedScore'] / 10)

    # 3. Ajuste por salud mental (a mejor salud, más tiempo)
    # El impacto es neutral si el puntaje es 5/10.
    mental_health_adjustment = 1.0 * ((datos['MentalHealthScore'] - 5) / 5)

    # 4. Cálculo final y clipping a un rango saludable
    datos["RecommendedScreenTime"] = (
        datos['BaselineHours'] + addiction_adjustment + mental_health_adjustment
    ).clip(0.5, 4).round(1)

    # --- Entrenamiento del modelo ---
    features = ["Age", "AddictedScore", "MentalHealthScore"]
    target = "RecommendedScreenTime"

    modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(datos[features], datos[target])

    joblib.dump(modelo, _MODELO_PATH)

    return modelo, datos


def CompletarRecommendedScreenTime(df):
    """Completa la columna RecommendedScreenTime usando el modelo predictivo."""
    modelo, _ = _get_or_train_model()

    mask = df["RecommendedScreenTime"].isna()
    if mask.any():
        X_pred = df.loc[mask, ["Age", "AddictedScore", "MentalHealthScore"]]
        df.loc[mask, "RecommendedScreenTime"] = modelo.predict(X_pred).round(1)

    return df


def PredecirRecommendedScreenTime(Age: int, AddictedScore: int, MentalHealth: int):
    """Predice horas recomendadas de uso y genera una gráfica explicativa."""
    Modelo, SinteticoDF = _get_or_train_model()

    # Mapeo de columnas para la visualización (por si acaso)
    for col_lower in ["age", "addicted_score", "mental_health_score", "recommended_screen_time"]:
        col_pascal = col_lower.title().replace("_", "")
        if col_pascal not in SinteticoDF.columns and col_lower in SinteticoDF.columns:
            SinteticoDF[col_pascal] = SinteticoDF[col_lower]
    
    # Si la columna no existe después del mapeo, la calcula para la gráfica
    if "RecommendedScreenTime" not in SinteticoDF.columns:
         SinteticoDF["RecommendedScreenTime"] = Modelo.predict(SinteticoDF[["Age", "AddictedScore", "MentalHealthScore"]]).round(1)


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
    # Para la línea de tendencia, usamos los valores del usuario para adicción y salud mental
    # para mostrar la recomendación específica para su perfil a través de las edades.
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

    # Calcular R^2 score para metadata
    features = ["Age", "AddictedScore", "MentalHealthScore"]
    target = "RecommendedScreenTime"
    r2_score = Modelo.score(SinteticoDF[features], SinteticoDF[target])


    return {
        "RecommendedHours": HorasPred,
        "Grafica": f"data:image/png;base64,{GraphBase64}",
        "Entrada": {
            "Age": Age,
            "AddictedScore": AddictedScore,
            "MentalHealthScore": MentalHealth,
        },
        "modelo_metadata": {
            "algoritmo": "RandomForest",
            "variables": ["age", "addicted_score", "mental_health_score"],
            "explicacion_modelo": {
                "que_es": "Es un modelo que usa muchos árboles de decisión para predecir cuántas horas de pantalla son recomendables.",
                "como_funciona": "Aprende de datos reales considerando tu edad, nivel de adicción y salud mental para dar una recomendación personalizada.",
                "para_que_sirve": "Te ayuda a saber cuánto tiempo de pantalla es saludable para ti."
            },
            "r2": float(r2_score)
        },
        "interpretacion_graficas": {
            "scatter": "Cada punto azul es una persona de la comunidad. La estrella roja eres tú. La línea negra muestra la recomendación para tu perfil."
        }
    } 
from utils.data_helpers.SleepQualityDataHelper import GenerarDatosSleepQuality
from utils.plot_utils import PlotToBase64
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from model_utils_excel import cargar_pkl_enriquecido
from pathlib import Path
import joblib

# Ruta para guardar/cargar el modelo entrenado
_MODELO_DIR = Path(__file__).resolve().parent.parent / "modelos" / "sleep_quality"
_MODELO_DIR.mkdir(parents=True, exist_ok=True)
_MODELO_PATH = _MODELO_DIR / "decision_tree.pkl"


def CompletarSleepQualityCategory(DataFrame):
    """Completa columna SleepQualityCategory si está vacía o contiene NaN."""
    SinteticoDF = GenerarDatosSleepQuality(6000)
    Modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
    Modelo.fit(
        SinteticoDF[["SleepHoursPerNight", "MentalHealthScore", "AddictedScore"]],
        SinteticoDF["SleepQualityCategory"],
    )

    Mask = DataFrame["SleepQualityCategory"].isna()
    if Mask.any():
        XPredict = DataFrame.loc[Mask, [
            "SleepHoursPerNight",
            "MentalHealthScore",
            "AddictedScore",
        ]]
        DataFrame.loc[Mask, "SleepQualityCategory"] = Modelo.predict(XPredict)

    return DataFrame


def PredecirSleepQuality(SleepHours: float, MentalHealth: int, AddictedScore: int):
    """Predice calidad de sueño y devuelve dict con gráfica base64 usando datos reales."""

    # Cargar o entrenar modelo
    if _MODELO_PATH.exists():
        Modelo = joblib.load(_MODELO_PATH)
        ComunidadDF = cargar_pkl_enriquecido()
    else:
        ComunidadDF = cargar_pkl_enriquecido()

    # Mapear nombres de columnas a PascalCase si es necesario (siempre)
    if "SleepHoursPerNight" not in ComunidadDF.columns and "sleep_hours_per_night" in ComunidadDF.columns:
        ComunidadDF["SleepHoursPerNight"] = ComunidadDF["sleep_hours_per_night"]
    if "MentalHealthScore" not in ComunidadDF.columns and "mental_health_score" in ComunidadDF.columns:
        ComunidadDF["MentalHealthScore"] = ComunidadDF["mental_health_score"]
    if "AddictedScore" not in ComunidadDF.columns and "addicted_score" in ComunidadDF.columns:
        ComunidadDF["AddictedScore"] = ComunidadDF["addicted_score"]

    # Crear la etiqueta si no existe
    if "SleepQualityCategory" not in ComunidadDF.columns:
        ComunidadDF["SleepQualityCategory"] = (
            ComunidadDF["SleepHoursPerNight"] >= 7
        ).astype(int)

    if not _MODELO_PATH.exists():
        Modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
        Modelo.fit(
            ComunidadDF[["SleepHoursPerNight", "MentalHealthScore", "AddictedScore"]],
            ComunidadDF["SleepQualityCategory"],
        )
        joblib.dump(Modelo, _MODELO_PATH)

    Prediccion = int(Modelo.predict([[SleepHours, MentalHealth, AddictedScore]])[0])

    # Crear gráfica de dona destacando la categoría del usuario usando datos reales
    Fig, Ax = plt.subplots(figsize=(6, 6))

    Conteo = (
        ComunidadDF["SleepQualityCategory"].value_counts()
        .reindex([0, 1])
        .fillna(0)
    )
    Labels = ["Mala", "Buena"]
    Colors = ["#e74c3c", "#2ecc71"]

    Total = Conteo.sum()
    Porcentajes = (Conteo / Total * 100).round(1)

    Explode = [0.1 if idx == Prediccion else 0 for idx in range(len(Labels))]

    Wedges, _ = Ax.pie(
        Conteo,
        labels=[f"{Labels[idx]} ({Porcentajes[idx]}%)" for idx in range(len(Labels))],
        colors=Colors,
        explode=Explode,
        startangle=90,
        textprops={"fontsize": 12},
    )

    Centro = plt.Circle((0, 0), 0.70, fc="white")
    Fig.gca().add_artist(Centro)
    Ax.set_aspect("equal")
    Ax.set_title("Calidad de Sueño: Comunidad vs. Tú")

    Ax.legend(
        [Wedges[Prediccion]],
        [f"Tú: {Labels[Prediccion]}"],
        loc="upper right",
        frameon=False,
    )

    GraficaBase64 = PlotToBase64(Fig)

    return {
        "Categoria": Labels[int(Prediccion)],
        "Grafica": f"data:image/png;base64,{GraficaBase64}",
        "Entrada": {
            "SleepHoursPerNight": SleepHours,
            "MentalHealthScore": MentalHealth,
            "AddictedScore": AddictedScore,
        },
        "modelo_metadata": {
            "algoritmo": "DecisionTreeClassifier",
            "variables": [
                "sleep_hours_per_night",
                "mental_health_score",
                "addicted_score",
            ],
            "explicacion_modelo": {
                "que_es": "Árbol de decisión que clasifica tu calidad de sueño en buena o mala basándose en horas de sueño y bienestar.",
                "como_funciona": "Divide el espacio de variables en reglas simples para llegar a la predicción de categoría.",
                "para_que_sirve": "Ofrecer una clasificación clara y accionable sobre tu descanso."
            },
            "modelo_persistente": True
        },
        "interpretacion_graficas": {
            "dona": "La dona muestra la proporción de la comunidad con sueño bueno (verde) y malo (rojo). Tu categoría está resaltada con un efecto de separación."
        }
    } 
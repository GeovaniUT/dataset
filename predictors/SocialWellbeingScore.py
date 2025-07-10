from utils.data_helpers.SocialWellbeingDataHelper import GenerarDatosSocialWellbeing
from utils.plot_utils import PlotToBase64
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def CompletarSocialWellbeingScore(df):
    sintetico = GenerarDatosSocialWellbeing(6000)
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(
        sintetico[["RelationshipStatus", "MentalHealthScore", "MostUsedPlatform", "AddictedScore"]],
        sintetico["SocialWellbeingScore"],
    )

    mask = df["SocialWellbeingScore"].isna()
    if mask.any():
        X_pred = df.loc[mask, [
            "RelationshipStatus",
            "MentalHealthScore",
            "MostUsedPlatform",
            "AddictedScore",
        ]]
        df.loc[mask, "SocialWellbeingScore"] = model.predict(X_pred).round().clip(1, 10)

    return df


def PredecirSocialWellbeingScore(relationship_status: int, mental_health: int, platform: int, addicted_score: int):
    sintetico = GenerarDatosSocialWellbeing(6000)
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(
        sintetico[["RelationshipStatus", "MentalHealthScore", "MostUsedPlatform", "AddictedScore"]],
        sintetico["SocialWellbeingScore"],
    )

    pred_score = int(model.predict([[relationship_status, mental_health, platform, addicted_score]])[0].round().clip(1, 10))

    # Barra individual comparada con promedio
    promedio = sintetico["SocialWellbeingScore"].mean()
    fig, ax = plt.subplots()
    ax.bar(["Promedio"], [promedio], color="#3498db", label="Promedio Comunidad")
    ax.bar(["Tú"], [pred_score], color="#e67e22", label="Tu Puntaje")
    ax.set_ylim(0, 10)
    ax.set_ylabel("Puntaje 1-10")
    ax.set_title("Bienestar Social Percibido")
    ax.legend()

    graph = PlotToBase64(fig)

    return {
        "SocialWellbeingScore": pred_score,
        "Grafica": f"data:image/png;base64,{graph}",
        "Entrada": {
            "RelationshipStatus": relationship_status,
            "MentalHealthScore": mental_health,
            "MostUsedPlatform": platform,
            "AddictedScore": addicted_score,
        },
        "ModeloMetadata": {
            "Algoritmo": "RandomForestRegressor",
            "Variables": [
                "RelationshipStatus",
                "MentalHealthScore",
                "MostUsedPlatform",
                "AddictedScore",
            ],
        },
    } 
import numpy as np
import pandas as pd


def GenerarDatosRecommendedScreenTime(N: int = 5000) -> pd.DataFrame:
    """Genera datos sintéticos para horas recomendadas de uso de pantalla."""
    rng = np.random.default_rng(45)

    Age = rng.integers(17, 41, N)  # 17 a 40 años
    AddictedScore = rng.integers(1, 11, N)
    MentalHealthScore = rng.integers(1, 11, N)

    # Fórmula heurística para horas recomendadas
    RecommendedHours = 2 + 3 * (AddictedScore / 10) - 1.5 * (MentalHealthScore / 10) + Age / 60
    RecommendedHours = np.clip(RecommendedHours, 0.5, 6).round(1)

    return pd.DataFrame({
        "Age": Age,
        "AddictedScore": AddictedScore,
        "MentalHealthScore": MentalHealthScore,
        "RecommendedScreenTime": RecommendedHours,
    }) 
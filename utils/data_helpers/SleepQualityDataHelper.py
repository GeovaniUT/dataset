import numpy as np
import pandas as pd


def GenerarDatosSleepQuality(N: int = 5000) -> pd.DataFrame:
    """Genera un DataFrame sintético para calidad de sueño."""
    RNG = np.random.default_rng(42)

    Sleep = RNG.integers(3, 10, N)
    MentalHealth = RNG.integers(1, 11, N)
    Addicted = RNG.integers(1, 11, N)

    SleepQualityCategory = (Sleep >= 7).astype(int)  # 1 Buena, 0 Mala

    return pd.DataFrame({
        "SleepHoursPerNight": Sleep,
        "MentalHealthScore": MentalHealth,
        "AddictedScore": Addicted,
        "SleepQualityCategory": SleepQualityCategory
    }) 
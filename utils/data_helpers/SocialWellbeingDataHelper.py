import numpy as np
import pandas as pd


def GenerarDatosSocialWellbeing(N: int = 5000) -> pd.DataFrame:
    """Genera datos sintéticos para SocialWellbeingScore (1-10)."""
    rng = np.random.default_rng(46)

    RelationshipStatus = rng.integers(1, 4, N)  # 1 Single, 2 InRelationship, 3 Complicated
    MentalHealthScore = rng.integers(1, 11, N)
    MostUsedPlatform = rng.integers(1, 5, N)
    AddictedScore = rng.integers(1, 11, N)

    # Heurística: mejor bienestar si mental alta y adicción baja
    base = MentalHealthScore * 0.6 + (11 - AddictedScore) * 0.3
    rel_adj = np.where(RelationshipStatus == 2, 1.0, -0.5)  # relación estable mejora
    score = base / 1.5 + rel_adj

    SocialWellbeingScore = np.clip(score, 1, 10).round().astype(int)

    return pd.DataFrame({
        "RelationshipStatus": RelationshipStatus,
        "MentalHealthScore": MentalHealthScore,
        "MostUsedPlatform": MostUsedPlatform,
        "AddictedScore": AddictedScore,
        "SocialWellbeingScore": SocialWellbeingScore,
    }) 
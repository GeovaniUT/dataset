import numpy as np
import pandas as pd


def GenerarDatosConflictRisk(N: int = 5000) -> pd.DataFrame:
    """Genera datos sintéticos para riesgo de conflictos en redes sociales."""
    rng = np.random.default_rng(44)

    AddictedScore = rng.integers(1, 11, N)
    RelationshipStatus = rng.integers(1, 4, N)
    AvgDailyUsageHours = rng.integers(1, 11, N)
    MostUsedPlatform = rng.integers(1, 10, N)  

    # Regla heurística de riesgo
    base_prob = 0.1 + 0.07 * (AddictedScore - 1)  # más adicción ⇒ más riesgo
    base_prob += 0.05 * (AvgDailyUsageHours > 5)
    base_prob += 0.05 * (RelationshipStatus == 3)
    base_prob = np.clip(base_prob, 0, 0.9)

    ConflictRisk = rng.binomial(1, base_prob)

    return pd.DataFrame({
        "AddictedScore": AddictedScore,
        "RelationshipStatus": RelationshipStatus,
        "AvgDailyUsageHours": AvgDailyUsageHours,
        "MostUsedPlatform": MostUsedPlatform,
        "ConflictRisk": ConflictRisk,
    }) 
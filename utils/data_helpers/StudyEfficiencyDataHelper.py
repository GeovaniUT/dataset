import numpy as np
import pandas as pd


def GenerarDatosStudyEfficiency(N: int = 5000) -> pd.DataFrame:
    """Genera datos sintéticos para StudyEfficiencyScore (0-100)."""
    rng = np.random.default_rng(47)

    AffectsAcademicPerformance = rng.binomial(1, 0.4, N)  # 40% percibe afectación
    AvgDailyUsageHours = rng.integers(1, 11, N)
    SleepHoursPerNight = rng.integers(3, 10, N)

    # Eficiencia base 90, resta por uso y afectación, suma por sueño
    score = 90 - AvgDailyUsageHours * 4 - AffectsAcademicPerformance * 15 + (SleepHoursPerNight - 5) * 3
    StudyEfficiencyScore = np.clip(score, 0, 100).round().astype(int)

    return pd.DataFrame({
        "AffectsAcademicPerformance": AffectsAcademicPerformance,
        "AvgDailyUsageHours": AvgDailyUsageHours,
        "SleepHoursPerNight": SleepHoursPerNight,
        "StudyEfficiencyScore": StudyEfficiencyScore,
    }) 
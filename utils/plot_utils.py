import base64
import io
import matplotlib.pyplot as plt


def PlotToBase64(Fig):
    """Convierte un objeto Figure de Matplotlib a una cadena base64 lista para JSON."""
    Buffer = io.BytesIO()
    Fig.savefig(Buffer, format="png", bbox_inches="tight")
    plt.close(Fig)
    Buffer.seek(0)
    return base64.b64encode(Buffer.read()).decode("utf-8") 
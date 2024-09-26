import io

from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def show_plot_mri(img, name=""):
    fig = Figure()
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap="gray")
    ax.set_title(f"MRI image {name}")

    buf = io.BytesIO()
    canvas.print_png(buf)
    qpm = QPixmap()
    qpm.loadFromData(buf.getvalue(), "PNG")
    return qpm

import io

import numpy as np
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def show_plot_eeg(data, name, channel_number):
    from app.properties.cnn_config import FS, ELECTRODE_POSITIONS

    fig = Figure()
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    t = np.arange(0, data[channel_number].shape[0]) / FS
    signal = data[channel_number]

    ax.plot(t, signal, label=f"Channel {channel_number}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sample values")
    ax.set_title(
        f"Signal plot {name}\nChannel: {ELECTRODE_POSITIONS[channel_number]}"
    )

    buf = io.BytesIO()
    canvas.print_png(buf)
    qpm = QPixmap()
    qpm.loadFromData(buf.getvalue(), "PNG")
    return qpm

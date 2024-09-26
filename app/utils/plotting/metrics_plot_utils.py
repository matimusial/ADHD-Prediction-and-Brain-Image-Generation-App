import io

from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def plot_metrics(controller_instance, metrics, plot_label):
    fig = Figure()
    canvas = FigureCanvas(fig)

    ax1 = fig.add_subplot(211)
    ax1.plot(
        range(1, len(metrics['accuracy']) + 1),
        metrics['accuracy'],
        'r-',
        label='Training Accuracy'
    )
    ax1.plot(
        range(1, len(metrics['val_accuracy']) + 1),
        metrics['val_accuracy'],
        'b-',
        label='Validation Accuracy'
    )
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(1, controller_instance.real_time_metrics.total_epochs)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = fig.add_subplot(212)
    ax2.plot(
        range(1, len(metrics['loss']) + 1),
        metrics['loss'],
        'r-',
        label='Training Loss'
    )
    ax2.plot(
        range(1, len(metrics['val_loss']) + 1),
        metrics['val_loss'],
        'b-',
        label='Validation Loss'
    )
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(1, controller_instance.real_time_metrics.total_epochs)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    buf = io.BytesIO()
    canvas.print_png(buf)
    qpm = QPixmap()
    qpm.loadFromData(buf.getvalue(), 'PNG')
    plot_label.setPixmap(qpm)
    buf.close()


def plot_metrics_gan(controller_instance, gan_metrics, metrics_label, generated_img_label):
    fig = Figure()
    canvas = FigureCanvas(fig)

    epoch_numbers = gan_metrics['epoch_number']

    ax1 = fig.add_subplot(211)
    ax1.plot(
        epoch_numbers,
        gan_metrics['train_d_loss'],
        'r-',
        label='Train D Loss'
    )
    ax1.plot(
        epoch_numbers,
        gan_metrics['val_d_loss'],
        'b-',
        label='Val D Loss'
    )
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Discriminator Loss')
    ax1.set_title('Discriminator Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(1, controller_instance.real_time_metrics.total_epochs)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = fig.add_subplot(212)
    ax2.plot(
        epoch_numbers,
        gan_metrics['train_g_loss'],
        'r-',
        label='Train G Loss'
    )
    ax2.plot(
        epoch_numbers,
        gan_metrics['val_g_loss'],
        'b-',
        label='Val G Loss'
    )
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Generator Loss')
    ax2.set_title('Generator Loss')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(1, controller_instance.real_time_metrics.total_epochs)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    buf = io.BytesIO()
    canvas.print_png(buf)
    qpm = QPixmap()
    qpm.loadFromData(buf.getvalue(), 'PNG')
    metrics_label.setPixmap(qpm)
    buf.close()
    generated_img_label.setPixmap(gan_metrics['img'])

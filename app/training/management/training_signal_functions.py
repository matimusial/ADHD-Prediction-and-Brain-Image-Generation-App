from app.utils.plotting.metrics_plot_utils import plot_metrics, plot_metrics_gan


def connect_signals(controller_instance, real_time_metrics):
    """
    Connects real-time metrics signal to the controller's GUI update function.
    """
    real_time_metrics.metrics_updated.connect(controller_instance.update_gui)
    controller_instance.real_time_metrics = real_time_metrics


def update_gui(controller_instance, epoch, metrics, progress_bar, plot_label):
    """
    Updates the GUI during training by adjusting the progress bar and plotting metrics.
    """
    progress_percentage = int(((epoch + 1) / controller_instance.real_time_metrics.total_epochs) * 100)
    progress_bar.setValue(progress_percentage)
    plot_metrics(controller_instance, metrics, plot_label)


def connect_signals_gan(controller_instance, real_time_metrics):
    """
    Connects real-time metrics signal to the controller's GAN-specific GUI update function..
    """
    real_time_metrics.metrics_updated.connect(controller_instance.update_gui_gan)
    controller_instance.real_time_metrics = real_time_metrics


def update_gui_gan(controller_instance, epoch, gan_metrics, progress_bar, metrics_label, generated_img_label):
    """
    Updates the GUI during GAN training by adjusting the progress bar and plotting metrics,
    also updates the label for the generated image.
    """
    progress_percentage = int(((epoch) / controller_instance.real_time_metrics.total_epochs) * 100)
    progress_bar.setValue(progress_percentage)
    plot_metrics_gan(controller_instance, gan_metrics, metrics_label, generated_img_label)


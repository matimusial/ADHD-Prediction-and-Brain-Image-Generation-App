from PyQt5.QtCore import QThread

from app.training.management.workers.admin_worker import AdminWorker
from app.training.management.workers.generate_new_worker import GenerateNewWorker
from app.training.management.workers.doctor_worker import DoctorWorker
from app.services.gui_services.alerts import handle_error


def create_admin_controller_thread(instance):
    thread = QThread()
    worker = AdminWorker(instance)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(instance.on_finished_runner)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    worker.error.connect(instance.on_error)
    instance.thread = thread
    instance.worker = worker
    thread.start()


def create_generate_new_controller_thread(instance):
    thread = QThread()
    worker = GenerateNewWorker(instance.db, instance.chosen_model_data[0])
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(instance.on_model_loaded)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    worker.error.connect(handle_error)
    instance.thread = thread
    instance.worker = worker
    thread.start()


def create_doctor_controller_thread(instance):
    thread = QThread()
    worker = DoctorWorker(instance)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(instance.on_finished)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    worker.error.connect(instance.on_error)
    instance.thread = thread
    instance.worker = worker
    thread.start()

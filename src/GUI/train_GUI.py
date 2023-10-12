import os
import sys
sys.path.append(".")
from turtle import update
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTextEdit,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QWidget,
    QComboBox
)
import PyQt5.QtGui as QtGui
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
import torch
from torch.utils.data import Dataset, DataLoader
import time
import datetime
import numpy as np
from PIL import Image
from tifffile import TiffFile
import tifffile
import skimage.io as skio
from src.utils.dataset_pyqt import DatasetSupport_test_stitch
import glob

from model.SUPPORT import SUPPORT
from src.utils.dataset_pyqt import DatasetSUPPORT
from src.utils.dataset import random_transform
from torch.utils.tensorboard import SummaryWriter


from PyQt5 import QtWidgets


class QHSeperationLine(QtWidgets.QFrame):
    '''
    a horizontal seperation line\n
    '''
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        return

class QVSeperationLine(QtWidgets.QFrame):
    '''
    a vertical seperation line\n
    '''
    def __init__(self):
        super().__init__()
        self.setFixedWidth(20)
        self.setMinimumHeight(1)
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        return

def convert_nparray_to_QPixmap(img):
    _, _, ch = img.shape
    if ch == 1:
        PIL_image = Image.fromarray(img[:, :, 0], mode="L")
        im2 = PIL_image.convert("RGBA")
        data = im2.tobytes("raw", "BGRA")
        qim = QtGui.QImage(
            data, PIL_image.width, PIL_image.height, QtGui.QImage.Format_ARGB32
        )
        qpixmap = QPixmap.fromImage(qim)

    else:
        raise RuntimeError

    return qpixmap

class modelThread(QThread):
    finish_loading = pyqtSignal(int)
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self._isRunning = True

    def run(self):
        if self._isRunning:
            model = SUPPORT(
                in_channels=61,
                mid_channels=[16, 32, 64, 128, 256],
                depth=5,
                blind_conv_channels=64,
                one_by_one_channels=[32, 16],
                last_layer_channels=[64, 32, 16],
                bs_size=self.parent.bs_size,
            )
            
        if self._isRunning:
            if self.parent.cuda:
                model = model.cuda()

            self.parent.model = model
            
        if self._isRunning:
            self.parent.optimizer = torch.optim.Adam(self.parent.model.parameters(), lr=5e-4)
        
        self.finish_loading.emit(1)

    def stop(self):
        self._isRunning = False
        self.quit()
        # self.wait(1000)



class set_patch_Thread(QThread):
    finish_loading = pyqtSignal(int)
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self._isRunning = True

    def run(self):
        self.parent.t = torch.cuda.get_device_properties(0).total_memory
        self.parent.r = torch.cuda.memory_reserved(0)
        self.parent.a = torch.cuda.memory_allocated(0)

        # TODO
        self.parent.patch_size = [61, 128, 128]
        self.parent.patch_interval = [1, 64, 64]
            
        self.finish_loading.emit(1)

    def stop(self):
        self._isRunning = False
        self.quit()


class gen_dset_Thread(QThread):
    finish_loading = pyqtSignal(int)
    imgone_loading = pyqtSignal(int)
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self._isRunning = True

    def run(self):
        # self.parent.Text_img_info.append("\nReading tif files... Press stop button to interrupt.\n")
        if self.parent.imgs_list is not None:
            noisy_images_train = []
            for i, noisy_data in enumerate(self.parent.imgs_list):
                if self._isRunning:
                    noisy_image = torch.from_numpy(skio.imread(noisy_data).astype(np.float32)).type(
                        torch.FloatTensor
                    )
                    T, _, _ = noisy_image.shape
                    noisy_images_train.append(noisy_image)
                    self.imgone_loading.emit(1)
                    # self.parent.Text_img_info.append(f". ")
                # self.parent.Text_img_info.append("\n")
            
            if self._isRunning:
                self.parent.dataset_train = DatasetSUPPORT(
                    noisy_images_train,
                    patch_size=self.parent.patch_size,
                    patch_interval=self.parent.patch_interval,
                    transform=None,
                    random_patch=True,
                )
                self.parent.dataloader_train = DataLoader(
                    self.parent.dataset_train,
                    batch_size=16,
                    shuffle=False
                )
            
        self.finish_loading.emit(1)

    def stop(self):
        self._isRunning = False
        self.quit()

class runThread(QThread):
    signal_update_img = pyqtSignal(int)
    progressbar_update = pyqtSignal(int)
    log_update = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self._isRunning = True
        self._ti = 0

    def run(self):
        while self.parent.model_loaded is False:
            pass

        start_time = time.time()
        counter = 0

        metadata = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{self.parent.save_header}/{metadata}", exist_ok=True)
        os.makedirs(f"{self.parent.save_header}/{metadata}/tsboard/", exist_ok=True)
        self.parent.save_path = f"{self.parent.save_header}/{metadata}"

        # save information
        with open(f"{self.parent.save_header}/{metadata}/img_info.txt", "w+") as f:
            f.writelines(self.parent.Text_img_info.toPlainText())
        with open(f"{self.parent.save_header}/{metadata}/model_info.txt", "w+") as f:
            f.writelines(self.parent.Text_model_info.toPlainText())

        writer = SummaryWriter(f"{self.parent.save_header}/{metadata}/tsboard/")

        noisy_image_eval, coord, idx = next(iter(self.parent.dataloader_train))
        noisy_image_eval = noisy_image_eval.cuda()
        # print(noisy_image.shape)
        rng = np.random.default_rng(0)
        
        for epoch in range(0, 10):
            self.parent.model.train()

            loss_list_l1 = []
            loss_list_l2 = []
            loss_list = []

            L1_pixelwise = torch.nn.L1Loss()
            L2_pixelwise = torch.nn.MSELoss()

            # training
            for i, data in enumerate(self.parent.dataloader_train):
                if self._isRunning:
                    self._ti += 1
                    (noisy_image, _, ds_idx) = data
                    noisy_image, _ = random_transform(noisy_image, None, rng)
                    
                    B, T, X, Y = noisy_image.shape
                    noisy_image = noisy_image.cuda()
                    noisy_image_target = torch.unsqueeze(noisy_image[:, int(T/2), :, :], dim=1)

                    noisy_image_denoised = self.parent.model(noisy_image)
                    loss_l1_pixelwise = L1_pixelwise(noisy_image_denoised, noisy_image_target)
                    loss_l2_pixelwise = L2_pixelwise(noisy_image_denoised, noisy_image_target)
                    loss_sum = 0.5 * loss_l1_pixelwise + 0.5 * loss_l2_pixelwise

                    self.parent.optimizer.zero_grad()
                    loss_sum.backward()
                    self.parent.optimizer.step()

                    loss_list_l1.append(loss_l1_pixelwise.item())
                    loss_list_l2.append(loss_l2_pixelwise.item())
                    loss_list.append(loss_sum.item())

                    # print log
                    if (epoch % 1 == 0) and (i % 50 == 0):
                        loss_mean = np.mean(np.array(loss_list))
                        loss_mean_l1 = np.mean(np.array(loss_list_l1))
                        loss_mean_l2 = np.mean(np.array(loss_list_l2))

                        # ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        writer.add_scalar("Loss_l1/train_batch", loss_mean, epoch*len(self.parent.dataloader_train) + i)
                        writer.add_scalar("Loss_l2/train_batch", loss_mean_l1, epoch*len(self.parent.dataloader_train) + i)
                        writer.add_scalar("Loss/train_batch", loss_mean_l2, epoch*len(self.parent.dataloader_train) + i)
                        
                        # logging.info(f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] Batch [{i+1}/{len(self.parent.dataloader_train)}] "+\
                        #     f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}")
                        print(f"{i} {epoch} loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f}")
                        self.log_update.emit(f"Epoch : {epoch} Iter : {i} Loss : {loss_mean:.3f}, loss_l1 : {loss_mean_l1:.3f}, loss_l2 : {loss_mean_l2:.3f}")

                    if (epoch % 1 == 0) and (i % 50 == 0):
                        noisy_image_eval_target = torch.unsqueeze(noisy_image_eval[:, int(T/2), :, :], dim=1)
                        noisy_image_eval_denoised = self.parent.model(noisy_image_eval)

                        raw_frame = noisy_image_eval_target * self.parent.dataset_train.std_images[0] + self.parent.dataset_train.mean_images[0]
                        denoised_frame = noisy_image_eval_denoised * self.parent.dataset_train.std_images[0] + self.parent.dataset_train.mean_images[0]

                        disp_raw = raw_frame.detach().cpu().numpy()[0, 0] # w, h
                        disp_denoised = denoised_frame.detach().cpu().numpy()[0, 0] # w, h

                        if epoch == 0:
                            vmin = np.min(disp_raw)
                            vmax = np.max(disp_raw)

                        self.parent.disp_raw = np.clip(
                            1.2 * (disp_raw - vmin) / (vmax - vmin), 0, 1
                        )
                        self.parent.disp_denoised = np.clip(
                            1.2 * (disp_denoised - vmin) / (vmax - vmin), 0, 1
                        )

                        self.signal_update_img.emit(1)

                    self.progressbar_update.emit(1)

    def stop(self):
        self._isRunning = False
        self.quit()
        self.wait(1000)


class SUPPORTGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Train SUPPORT")
        # Create an outer layout
        widget = QWidget()
        widget.setStyleSheet("background-color: #F0F0F0;") # #ECECEC;")

        Layout_total = QGridLayout(widget)
        Layout_upper = QGridLayout()

        Layout_upper_pathinfo = QGridLayout()
        Layout_upper_dispimg = QVBoxLayout()

        #### Layout_total --> Layout_upper --> Layout_upper_pathinfo --> Layout_image_path
        Layout_image_path = QGridLayout()
        self.Label_image = QLabel("Train directory")
        self.Label_image.setMaximumWidth(250)
        self.Label_image.setStyleSheet("background-color: #FFFFFF; color: #444444")
        Layout_image_path.addWidget(self.Label_image, 0, 0, 1, 3)

        browse_icon = QIcon("./src/GUI/icons/browse.png")
        Button_browse_img = QPushButton()
        Button_browse_img.setIcon(browse_icon)
        Button_browse_img.clicked.connect(self.browse_img)
        Layout_image_path.addWidget(Button_browse_img, 0, 4, 1, 1)

        Layout_upper_pathinfo.addLayout(Layout_image_path, 0, 0, 1, 1)

        #### Layout_total --> Layout_upper --> Layout_upper_pathinfo --> Layout_image_info
        Layout_image_info = QGridLayout()
        self.Text_img_info = QTextEdit()
        self.Text_img_info.setText("Image information (Browse directory containing images)")
        self.Text_img_info.setReadOnly(True)
        Layout_image_info.addWidget(self.Text_img_info, 0, 0, 3, 4)

        """
        Layout_image_info.addWidget(QLabel("start index"), 3, 0, 1, 1)
        self.Text_start_idx = QLineEdit()
        self.Text_start_idx.textChanged.connect(self.change_start_idx)
        self.Text_start_idx.setStyleSheet("background-color: #FFFFFF; color: #444444")
        Layout_image_info.addWidget(self.Text_start_idx, 3, 1, 1, 1)
        Layout_image_info.addWidget(QLabel("end index"), 3, 2, 1, 1)
        self.Text_end_idx = QLineEdit()
        self.Text_end_idx.textChanged.connect(self.change_end_idx)
        self.Text_end_idx.setStyleSheet("background-color: #FFFFFF; color: #444444")
        Layout_image_info.addWidget(self.Text_end_idx, 3, 3, 1, 1)
        """
        Layout_upper_pathinfo.addLayout(Layout_image_info, 0, 1, 1, 2)

        #### Layout_total --> Layout_upper --> Layout_upper_pathinfo --> Layout_model
        Layout_model = QGridLayout()
        Combo_model = QComboBox(self)
        Combo_model.addItem("1")
        Combo_model.addItem("3")
        Combo_model.addItem("5")
        Combo_model.activated[str].connect(self.set_model)

        Layout_model.addWidget(QLabel("Blind spot size"))
        Layout_model.addWidget(Combo_model, 0, 4, 1, 1)

        """
        Layout_model.addWidget(QLabel("CPU/GPU"))
        self.Combo_cpu = QComboBox(self)
        self.Combo_cpu.addItem("CPU")
        self.Combo_cpu.addItem("GPU")
        self.Combo_cpu.activated[str].connect(self.set_cpugpu)
        Layout_model.addWidget(self.Combo_cpu, 1, 4, 1, 1)
        """

        """
        Layout_model.addWidget(QHSeperationLine(), 2, 0, 1, 5)
        
        Layout_load_model = QGridLayout()
        self.Label_custom_model = QLabel("Load custom model")
        self.Label_custom_model.setMaximumWidth(250)
        self.Label_custom_model.setStyleSheet("background-color: #FFFFFF; color: #444444")
        Layout_load_model.addWidget(self.Label_custom_model, 0, 0, 1, 3)

        browse_icon = QIcon("./src/GUI/icons/browse.png")
        Button_browse_img = QPushButton()
        Button_browse_img.setIcon(browse_icon)
        Button_browse_img.clicked.connect(self.browse_model)
        Layout_load_model.addWidget(Button_browse_img, 0, 3, 1, 1)

        Layout_model.addLayout(Layout_load_model, 3, 0, 1, 5)

        """
        Layout_upper_pathinfo.addLayout(Layout_model, 1, 0, 1, 1)

        self.Text_model_info = QTextEdit("Model information")
        self.Text_model_info.setReadOnly(True)
        Layout_upper_pathinfo.addWidget(self.Text_model_info, 1, 1, 1, 2)

        #### Layout_total --> Layout_upper --> Layout_upper_pathinfo --> Layout_model
        Layout_save = QGridLayout()
        self.savepath_path = QLabel("Save path")
        self.savepath_path.setMaximumWidth(250)
        self.savepath_path.setStyleSheet("background-color: #FFFFFF; color: #444444")
        Layout_save.addWidget(self.savepath_path, 0, 0, 1, 3)

        Button_browse_save = QPushButton()
        Button_browse_save.setIcon(browse_icon)
        Button_browse_save.clicked.connect(self.set_save_header)
        Layout_save.addWidget(Button_browse_save, 0, 4, 1, 1)

        Layout_upper_pathinfo.addLayout(Layout_save, 2, 0, 1, 1)

        self.Text_save = QTextEdit("Saving information")
        self.Text_save.setReadOnly(True)
        Layout_upper_pathinfo.addWidget(self.Text_save, 2, 1, 1, 2)

        #### Layout_total --> Layout_upper --> Layout_upper_dispimg --> Layout_label_fps
        Layout_label_fps = QHBoxLayout()
        Layout_label_fps.addWidget(QLabel("Raw image"))
        self.Label_fps = QLabel("fps : 00.00")
        self.Label_fps.setAlignment(Qt.AlignRight)
        Layout_label_fps.addWidget(self.Label_fps)

        Layout_upper_dispimg.addLayout(Layout_label_fps)

        raw_img = convert_nparray_to_QPixmap(np.zeros((128, 128, 1), dtype=np.uint8))
        raw_img = raw_img.scaled(QSize(300, 300), aspectRatioMode=Qt.KeepAspectRatio)
        self.Label_raw_img = QLabel()
        self.Label_raw_img.setPixmap(raw_img)
        Layout_upper_dispimg.addWidget(self.Label_raw_img)

        Layout_upper_dispimg.addWidget(QLabel("Denoised image"))

        denoised_img = convert_nparray_to_QPixmap(
            np.zeros((128, 128, 1), dtype=np.uint8)
        )
        denoised_img = denoised_img.scaled(
            QSize(300, 300), aspectRatioMode=Qt.KeepAspectRatio
        )
        self.Label_denoised_img = QLabel()
        self.Label_denoised_img.setPixmap(denoised_img)
        Layout_upper_dispimg.addWidget(self.Label_denoised_img)

        Layout_upper.addLayout(Layout_upper_pathinfo, 0, 0, 1, 3)
        Layout_upper.addLayout(Layout_upper_dispimg, 0, 3, 1, 1)

        #### Layout_total --> Layout_log
        Layout_log = QGridLayout()

        ## Just label
        Layout_log.addWidget(QLabel("Logs"), 0, 0, 1, 1)

        ## Actual logging board
        self.Text_log = QTextEdit()
        self.Text_log.setReadOnly(True)
        self.Text_log.setStyleSheet("background-color: #FFFFFF;")

        Layout_log.addWidget(self.Text_log, 1, 0, 3, 1)

        self.Button_run = QPushButton("Run")
        run_icon = QIcon("./src/GUI/icons/play.png")
        self.Button_run.setIcon(run_icon)
        self.Button_run.setIconSize(QSize(20, 20))
        self.Button_run.clicked.connect(self.run_SUPPORT)

        Layout_log.addWidget(self.Button_run, 1, 4, 1, 3)

        self.Button_stop = QPushButton("Stop")
        run_icon = QIcon("./src/GUI/icons/stop.png")
        self.Button_stop.setIcon(run_icon)
        self.Button_stop.setIconSize(QSize(20, 20))
        self.Button_stop.clicked.connect(self.stop_SUPPORT)
        self.Button_stop.setEnabled(False)
        Layout_log.addWidget(self.Button_stop, 2, 4, 1, 3)

        self.Progressbar = QProgressBar(self)
        self.Progressbar.setRange(0, 100)
        self.Progressbar.setValue(0)
        Layout_log.addWidget(self.Progressbar, 3, 4, 1, 3)

        Layout_total.addLayout(Layout_upper, 0, 0, 3, 1)
        # Layout_total.addWidget(QHSeperationLine(), 3, 0, 1, 1)
        Layout_total.addLayout(Layout_log, 4, 0, 1, 1)

        self.setCentralWidget(widget)
        self.setGeometry(100, 100, 1200, 800)

        self.setWindowIcon(QIcon("./src/GUI/icons/NICALab.png"))

        if torch.cuda.is_available():
            self.cuda = True
            cuda_available = "torch.CUDA is available. Will use GPU!"
            # self.Combo_cpu.setCurrentText("GPU")
        else:
            self.cuda = False
            cuda_available = "torch.CUDA is unavailable. Will not use GPU!"
            # self.Combo_cpu.setCurrentText("CPU")
        
        self.modelThr = None
        self.custom_model = False
        self.set_model("1")

        self.Text_log.append(cuda_available)

        self.show()

        self.model = None
        self.dir_path = None
        self.save_header = None
        self.thread = None
        
        self.patchThr = set_patch_Thread(self)
        self.start_patch_loading()
        self.patchThr.finish_loading.connect(self.finish_patch_loading)
        self.patchThr.start()        

    def start_patch_loading(self):
        self.patch_loaded = False

    def finish_patch_loading(self):
        self.Text_log.append(f"\ntotal memory (GB) : {self.t/(1024**3)}\nallocated memory (GB): {self.a/(1024**3)}")
        self.patch_loaded = True
        self.patchThr = None

    def start_model_loading(self):
        self.model_loaded = False

    def finish_model_loading(self):
        self.Text_log.append("\nFinish model loading.")
        self.model_loaded = True
        self.modelThr = None

    """
    def set_cpugpu(self, str):
        if str == "CPU":
            self.cuda = False
            self.Text_log.append("Selected CPU mode.")
            if self.model is not None:
                self.model = self.model.cpu()
            torch.cuda.empty_cache()
        elif str == "GPU":
            self.cuda = True
            self.Text_log.append("Selected GPU mode.")
            if self.model is not None:
                self.model = self.model.cuda()
        self.update_model_info()
    """

    def stop_SUPPORT(self):
        if self.thread is not None:
            self.thread.stop()
            self.onFinished()

    def set_model(self, str):
        if self.modelThr is not None:
            self.modelThr.stop()

        if str == "1":
            self.bs_size = 1
        elif str == "3":
            self.bs_size = 3
        elif str == "5":
            self.bs_size = 5
        self.update_model_info()
        
        self.modelThr = modelThread(self)
        self.start_model_loading()
        self.modelThr.finish_loading.connect(self.finish_model_loading)
        self.modelThr.start()
    
    def update_model_info(self):
        device = "GPU" if self.cuda else "CPU"
        custom_flag = "[CUSTOM MODEL, auto selecting bs_size] " if self.custom_model else ""
        self.Text_model_info.setText(
                f"{custom_flag}Blind spot size {self.bs_size} selected.\n\nWill be run on {device}."
            )

    def warning(self, message):
        QMessageBox.warning(self, "Warning", message)

    def onFinished(self):
        self.Button_run.setEnabled(True)
        self.Button_stop.setEnabled(False)

    def run_SUPPORT(self):
        # https://stackoverflow.com/questions/65276136/disable-elements-when-running-qthread-with-pyside-pyqt
        self.Button_run.setEnabled(False)
        self.Button_stop.setEnabled(True)

        if self.dir_path is None:
            self.warning("there is no image.")
            self.onFinished()
            return True

        """
        if self.Text_start_idx.text().isdigit() is False:
            self.warning("Check starting index. It is not digit.")
            self.onFinished()
            return True

        if self.Text_end_idx.text().isdigit() is False:
            self.warning("Check final index. It is not digit.")
            self.onFinished()
            return True

        if int(self.Text_start_idx.text()) < 1:
            self.warning(f"Starting index is smaller than 1.")
            self.onFinished()
            return True

        if int(self.Text_end_idx.text()) > self.img_t:
            self.warning(f"Final index is larger than image size {self.img_t}.")
            self.onFinished()
            return True
        """

        self.thread = runThread(self)
        self.startTraining()
        self.thread.signal_update_img.connect(self.update_img)
        self.thread.progressbar_update.connect(self.update_pbar_and_save)
        self.thread.log_update.connect(self.append_log)
        self.thread.finished.connect(self.onFinished)
        self.thread.start()

    def startTraining(self):
        self.Text_log.append(f"Number of training batch : {len(self.dataloader_train)}\nNumber of epochs : 10")

    def append_log(self, str):
        self.Text_log.append(str)

    def update_pbar_and_save(self):
        self.Progressbar.setValue(
            int(
                (self.thread._ti - 0 + 1)
                / (len(self.dataloader_train) * 10)
                * 100
            )
        )
        if self.thread._ti % len(self.dataloader_train) == 0:
            epoch = self.thread._ti // len(self.dataloader_train)
            
            torch.save(self.model.state_dict(), f"{self.save_path}/model_{epoch}.pth")
            torch.save(self.optimizer.state_dict(), f"{self.save_path}/optimizer_{epoch}.pth")
            self.Text_log.append(f"Model and Optimizer save, epoch : {epoch}")

    def update_img_init(self):
        raw_img = convert_nparray_to_QPixmap(
            (np.expand_dims(self.disp_raw, axis=2) * 255).astype(np.uint8)
        )
        raw_img = raw_img.scaled(QSize(300, 300), aspectRatioMode=Qt.KeepAspectRatio)
        self.Label_raw_img.setPixmap(raw_img)

    def update_img(self):
        raw_img = convert_nparray_to_QPixmap(
            (np.expand_dims(self.disp_raw, axis=2) * 255).astype(np.uint8)
        )
        raw_img = raw_img.scaled(QSize(300, 300), aspectRatioMode=Qt.KeepAspectRatio)
        self.Label_raw_img.setPixmap(raw_img)

        denoised_img = convert_nparray_to_QPixmap(
            (np.expand_dims(self.disp_denoised, axis=2) * 255).astype(np.uint8)
        )
        denoised_img = denoised_img.scaled(
            QSize(300, 300), aspectRatioMode=Qt.KeepAspectRatio
        )
        self.Label_denoised_img.setPixmap(denoised_img)

    def summarize_model(self, model):
        text = "\n"
        try:
            text += f"in_channels : {model.in_channels}\n"
            text += f"out_channels : {model.out_channels}\n"
            text += f"mid_channels : {model.mid_channels}\n"
            text += f"depth : {model.depth}\n"
            text += f"depth3x3 : {model.depth3x3}\n"
            text += f"depth5x5 : {model.depth5x5}\n"

            text += f"basic_conv_channels : {model.basic_conv_channels}\n"
            text += f"blind_conv_channels : {model.blind_conv_channels}\n"
            text += f"one_by_one_channels : {model.one_by_one_channels}\n"

            text += f"last_layer_channels : {model.last_layer_channels}\n"

            text += f"bs_size : {model.bs_size}\n"
        except:
            text = "Check model!"
        return text

    def set_save_header(self):
        fname = QFileDialog.getExistingDirectory(self, "Select folder", "./")
        # if os.path.isdir(fname):
        #     pass
        self.save_header = fname
        self.savepath_path.setText(
                f"{self.save_header}/YYYYMMDD_HHMMSS/denoised.tif"
            )
        self.Text_save.setText(
            f"Save as : {self.save_header}/YYYYMMDD_HHMMSS/denoised.tif"
        )

    def change_start_idx(self, text):
        if text.isdigit():
            self.actual_start_idx = int(text)
        else:
            pass
            # self.Text_log.append("Enter integer for starting index.")

    def change_end_idx(self, text):
        if text.isdigit():
            self.actual_end_idx = int(text)
        else:
            pass
            # self.Text_log.append("Enter integer for final index.")


    def browse_img(self):
        fname = QFileDialog.getExistingDirectory(self, "Select folder", "./")
        
        if fname is not None:
            self.Label_image.setText(fname)
            self.dir_path = fname

            self.imgs_list = glob.glob(f"{self.dir_path}/*.tif") # list(os.walk(self.dir_path, topdown=False))[-1][-1]
            self.num = len(self.imgs_list)

            print(self.dir_path)
            print(self.imgs_list)

            # find only tif files

            self.Text_img_info.setText(
                f"Image information (Browse directory containing images)\n\nSelected path : {self.dir_path}\nNumber of *.tif files : {self.num}\nFirst patch from first data ({self.imgs_list[0]}) will be presented at right side."
            )

            self.dsetThr = gen_dset_Thread(self)
            self.start_dset_loading()
            self.dsetThr.imgone_loading.connect(self.one_dset_loading)
            self.dsetThr.finish_loading.connect(self.finish_dset_loading)
            self.dsetThr.start()
        
            self.save_header = "/".join(fname.split("/")[:-1])
            self.save_header += "/SUPPORT_train"
            os.makedirs(f"{self.save_header}", exist_ok=True)

            self.Text_log.append(
                f"Default saving directory created : {self.save_header}"
            )
            self.Text_save.setText(
                f"(Default) Save as : {self.save_header}/YYYYMMDD_HHMMSS/denoised.tif\n\nYou can change it using Browse button."
            )
            """
            self.Text_save.setTextColor(QColor(255, 0, 0))
            self.Text_save.append(
                "\n\nWARNING : First 30 and last 30 denoised slices may not accurate result.\nRefer github.com/SUPPORT/issues/X."
            )
            self.Text_save.setTextColor(QColor(0, 0, 0))
            """

            self.savepath_path.setText(
                f"{self.save_header}/YYYYMMDD_HHMMSS/denoised.tif"
            )
            if self.model is None:
                self.set_model("1")
        

        else:
            pass

    def start_dset_loading(self):
        self.Text_img_info.append("\nStart loading dataset.")
        self.dset_loaded = False
    
    def one_dset_loading(self):
        self.Text_img_info.append(". ")

    def finish_dset_loading(self):
        self.Text_img_info.append("Finish loading dataset.")
        self.Text_log.append("\nLoaded dataset")
        self.dset_loaded = True
        self.dsetThr = None
        

        
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SUPPORTGUI()
    sys.exit(app.exec_())

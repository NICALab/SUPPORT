import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
import time
import datetime
import numpy as np
from PIL import Image
from tifffile import TiffFile
import tifffile
from src.utils.dataset_pyqt import DatasetSupport_test_stitch

from model.SUPPORT import SUPPORT


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
        if self.parent.model_path is not None:
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

            if self._isRunning:
                if self.parent.cuda:
                    state = torch.load(self.parent.model_path)
                else:
                    state = torch.load(self.parent.model_path, map_location="cpu")

            if self._isRunning:
                model.load_state_dict(state)

            self.parent.model = model
            
            self.finish_loading.emit(1)

    def stop(self):
        self._isRunning = False
        self.quit()
        # self.wait(1000)

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
        # save information
        with open(f"{self.parent.save_header}/{metadata}/img_info.txt", "w+") as f:
            f.writelines(self.parent.Text_img_info.toPlainText())
        with open(f"{self.parent.save_header}/{metadata}/model_info.txt", "w+") as f:
            f.writelines(self.parent.Text_model_info.toPlainText())

        img_one = None
        img_one_tensor = torch.zeros(61, self.parent.img_w, self.parent.img_h)

        memmap_image = tifffile.memmap(
            f"{self.parent.save_header}/{metadata}/denoised.tif",
            shape = self.parent.img_shape,
            dtype="float32"
        )

        with torch.no_grad():
            (t, _, _) = self.parent.img_shape
            self.parent.model.eval()

            # for ti in range(t):
            for ti in range(
                self.parent.actual_start_idx - 1, self.parent.actual_end_idx
            ):
                if self._isRunning == False:
                    break
                self._ti = ti

                counter += 1
                if (time.time() - start_time) > 1:
                    fps = counter / (time.time() - start_time)
                    self.parent.Label_fps.setText(f"fps : {fps:2.2f}")
                    counter = 0
                    start_time = time.time()

                start_idx = max(ti - 30, 0)
                end_idx = min(ti + 31, t)
                # print(ti)

                pad_front = max(30 - ti, 0)
                pad_end = max(31 - (t - ti), 0)
                # print(pad_front, pad_end)

                img_one = tifffile.imread(
                    self.parent.img_path, key=range(start_idx, end_idx, 1)
                )
                img_one_tensor[pad_front : 61 - pad_end, :, :] = torch.from_numpy(
                    img_one.astype(np.float32)
                ).type(torch.FloatTensor)
                if pad_front >= 1:
                    img_one_tensor[0:pad_front, :, :] = img_one_tensor[
                        pad_front
                    ].repeat(pad_front, 1, 1)
                if pad_end >= 1:
                    img_one_tensor[61 - pad_end :, :, :] = img_one_tensor[
                        61 - pad_end - 1
                    ].repeat(pad_end, 1, 1)

                if ti == self.parent.actual_start_idx - 1:
                    mean_one = torch.mean(img_one_tensor).item()
                    std_one = torch.std(img_one_tensor).item()

                w = 128 if self.parent.img_w > 128 else self.parent.img_w
                h = 128 if self.parent.img_h > 128 else self.parent.img_h

                testset = DatasetSupport_test_stitch(
                    img_one_tensor.clone().detach(),
                    patch_size=[61, w, h],
                    patch_interval=[1, w // 2, h // 2],
                    mean_image=mean_one,
                    std_image=std_one,
                )
                testloader = torch.utils.data.DataLoader(testset, batch_size=4)
                self.test_dataloader = testloader

                denoised_frame = np.zeros(
                    (1, *self.test_dataloader.dataset.noisy_image.shape[1:]),
                    dtype=np.float32,
                )

                for _, (noisy_image, _, single_coordinate) in enumerate(
                    self.test_dataloader
                ):
                    if self.parent.cuda:
                        noisy_image = noisy_image.cuda()  # [b, z, y, x]

                    noisy_image_denoised = self.parent.model(noisy_image)

                    for bi in range(noisy_image.size(0)):
                        stack_start_w = int(single_coordinate["stack_start_w"][bi])
                        stack_end_w = int(single_coordinate["stack_end_w"][bi])
                        patch_start_w = int(single_coordinate["patch_start_w"][bi])
                        patch_end_w = int(single_coordinate["patch_end_w"][bi])

                        stack_start_h = int(single_coordinate["stack_start_h"][bi])
                        stack_end_h = int(single_coordinate["stack_end_h"][bi])
                        patch_start_h = int(single_coordinate["patch_start_h"][bi])
                        patch_end_h = int(single_coordinate["patch_end_h"][bi])


                        denoised_frame[
                            0,  # stack_start_s,
                            stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w,
                        ] = (
                            noisy_image_denoised[bi]
                            .squeeze()[
                                patch_start_h:patch_end_h, patch_start_w:patch_end_w
                            ]
                            .cpu()
                        )

                denoised_frame = denoised_frame * std_one + mean_one

                disp_raw = img_one_tensor[30].detach().numpy()
                disp_denoised = denoised_frame[0]

                if ti == self.parent.actual_start_idx - 1:
                    vmin = np.min(disp_raw)
                    vmax = np.percentile(disp_raw, q=99) # np.max(disp_raw)

                self.parent.disp_raw = np.clip(
                    1.2 * (disp_raw - vmin) / (vmax - vmin), 0, 1
                )
                self.parent.disp_denoised = np.clip(
                    1.2 * (disp_denoised - vmin) / (vmax - vmin), 0, 1
                )

                self.signal_update_img.emit(1)
                self.progressbar_update.emit(1)

                memmap_image[ti, :, :] = denoised_frame
                
                # if ti == self.parent.actual_start_idx - 1:
                    # tifffile.imwrite(
                    #     f"{self.parent.save_header}/{metadata}/denoised.tif",
                    #     denoised_frame,
                    #     dtype="float32",
                    #     metadata={'axes': 'TYX', 'imagej_metadata': self.parent.imagej_metadata, }
                    # )
                # else:
                    # tifffile.imwrite(
                    #     f"{self.parent.save_header}/{metadata}/denoised.tif",
                    #     denoised_frame,
                    #     append=True,
                    #     dtype="float32",
                    #     metadata={'axes': 'TYX', 'imagej_metadata': self.parent.imagej_metadata}
                    # )
            
            memmap_image.flush()
            del memmap_image

    def stop(self):
        self._isRunning = False
        self.quit()
        self.wait(1000)


class SUPPORTGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SUPPORT denoise")
        # Create an outer layout
        widget = QWidget()
        widget.setStyleSheet("background-color: #ECECEC;")

        Layout_total = QGridLayout(widget)
        Layout_upper = QGridLayout()

        Layout_upper_pathinfo = QGridLayout()
        Layout_upper_dispimg = QVBoxLayout()

        #### Layout_total --> Layout_upper --> Layout_upper_pathinfo --> Layout_image_path
        Layout_image_path = QGridLayout()
        self.Label_image = QLabel("Image path")
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
        self.Text_img_info = QTextEdit("Image information")
        self.Text_img_info.setReadOnly(True)
        Layout_image_info.addWidget(self.Text_img_info, 0, 0, 3, 4)

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

        Layout_upper_pathinfo.addLayout(Layout_image_info, 0, 1, 1, 2)

        #### Layout_total --> Layout_upper --> Layout_upper_pathinfo --> Layout_model
        Layout_model = QGridLayout()
        Combo_model = QComboBox(self)
        Combo_model.addItem("1")
        Combo_model.addItem("3")
        Combo_model.addItem("5")
        Combo_model.activated[str].connect(self.set_model_path)

        Layout_model.addWidget(QLabel("Blind spot size"))
        Layout_model.addWidget(Combo_model, 0, 4, 1, 1)

        Layout_model.addWidget(QLabel("CPU/GPU"))
        self.Combo_cpu = QComboBox(self)
        self.Combo_cpu.addItem("CPU")
        self.Combo_cpu.addItem("GPU")
        self.Combo_cpu.activated[str].connect(self.set_cpugpu)
        Layout_model.addWidget(self.Combo_cpu, 1, 4, 1, 1)

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
            self.Combo_cpu.setCurrentText("GPU")
        else:
            self.cuda = False
            cuda_available = "torch.CUDA is unavailable. Will not use GPU!"
            self.Combo_cpu.setCurrentText("CPU")
        
        self.modelThr = None
        self.custom_model = False
        self.set_model_path("1")

        self.Text_log.append(cuda_available)

        self.show()

        self.model = None
        self.img_path = None
        self.save_header = None
        self.thread = None

    def start_model_loading(self):
        self.model_loaded = False

    def finish_model_loading(self):
        self.model_loaded = True
        self.modelThr = None

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

    def stop_SUPPORT(self):
        if self.thread is not None:
            self.thread.stop()
            self.onFinished()

    def set_model_path(self, str):
        if self.modelThr is not None:
            self.modelThr.stop()

        if str == "1":
            self.model_path = "./src/GUI/trained_models/bs1.pth"
            self.bs_size = 1
        elif str == "3":
            self.model_path = "./src/GUI/trained_models/bs3.pth"
            self.bs_size = 3
        elif str == "5":
            self.model_path = None
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
                f"{custom_flag} Blind spot size {self.bs_size} selected.\nModel path : {self.model_path}\n\nWill be run on {device}.\n\n\nYou can use custom model rather than built-ins.\nUse browse button to select custom model."
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

        if self.img_path is None:
            self.warning("there is no image.")
            self.onFinished()
            return True

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

        self.thread = runThread(self)
        self.thread.signal_update_img.connect(self.update_img)
        self.thread.progressbar_update.connect(self.update_pbar)
        self.thread.log_update.connect(self.append_log)
        self.thread.finished.connect(self.onFinished)
        self.thread.start()

    def append_log(self, str):
        self.Text_log.append(str)

    def update_pbar(self):
        self.Progressbar.setValue(
            int(
                (self.thread._ti - self.actual_start_idx + 1)
                / (self.actual_end_idx - self.actual_start_idx)
                * 100
            )
        )

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


    def browse_model(self):
        fname = QFileDialog.getOpenFileName(self, "(Load image) Open .pt or .pth file", "./")
        if fname[0].split(".")[-1] not in ["pt", "pth"]:
            self.warning("Select .pt or .pth file!")
            return True
        else:
            try:
                self.custom_model = True
                self.Label_custom_model.setText(fname[0])
                self.model_path = fname[0]
                
                state = torch.load(fname[0])
                blind_conv_channels = state["blind_convs3x3.0.weight"].size(0)
                out_conv0_channels = state["out_convs.0.weight"].size(1)

                depth3, depth5 = 0, 0
                for k in state:
                    if "blind_convs3x3" in k:
                        depth3 += 1
                    if "blind_convs5x5" in k:
                        depth5 += 1
                depth3 = depth3 // 3
                depth5 = depth5 // 3
                
                if (depth3 + depth5) == 2:
                    self.Text_log.append(f"[Warning] Model ambiguity in loading custom model. Please check the model.")

                if out_conv0_channels == 2 * blind_conv_channels:
                    self.bs_size = 3
                    self.Text_log.append(f"[Warning] Automatically set as bs_size = 3. Please check if it is right.")
                
                if out_conv0_channels == (depth3 + depth5) * blind_conv_channels:
                    self.bs_size = 1
                    self.Text_log.append(f"[Warning] Automatically set as bs_size = 1. Please check if it is right.")
                
                self.update_model_info()

                self.modelThr = modelThread(self)
                self.start_model_loading()
                self.modelThr.finish_loading.connect(self.finish_model_loading)
                self.modelThr.start()
    
            except:
                self.Label_image.setText("[ERROR] Please check logs.")
                self.img_path = None
                self.Text_log.append(
                    f"[ERROR]"
                )


    def browse_img(self):
        fname = QFileDialog.getOpenFileName(self, "(Load image) Open .tif file", "./")
        if fname[0].split(".")[-1] not in ["tif", "hdf5"]:
            self.warning("Select .tif file!")
            return True
        else:
            if fname[0].split(".")[-1] == "tif":
                self.Label_image.setText(fname[0])
                self.img_path = fname[0]
                tif = TiffFile(fname[0])
                print(tif)
                print(vars(tif))
                self.imagej_metadata = tif.imagej_metadata
                try:
                    series = tif.series[0]
                    t, w, h = series.shape
                    self.img_t = t
                    self.img_w = w
                    self.img_h = h
                    self.img_shape = series.shape
                    dtype = series.dtype
                    axes = series.axes
                    tif.close()

                    self.Text_img_info.setText(
                        f"Data name : {fname[0]}\nShape : ({w}, {h}, {t}) in (w, h, t) order.\ndtype : {dtype}\naxes : {axes}"
                    )
                    self.Text_start_idx.setText("1")
                    self.Text_end_idx.setText(f"{t}")
                    self.actual_start_idx = 1
                    self.actual_end_idx = t

                    raw_first = tifffile.imread(fname[0], key=0)
                    vmin = np.min(raw_first)
                    vmax = np.max(raw_first)

                    self.disp_raw = np.clip(
                        1.2 * (raw_first - vmin) / (vmax - vmin), 0, 1
                    )
                    self.update_img_init()

                except:
                    self.Label_image.setText("[ERROR] Please check logs.")
                    self.img_path = None
                    self.Text_log.append(
                        f"[ERROR] selected tif file seems not a stacked image. Please check file."
                    )

            elif fname[0].split(".")[-1] == "hdf5":
                pass

            self.save_header = "/".join(fname[0].split("/")[:-1])
            self.save_header += "/SUPPORT"
            os.makedirs(f"{self.save_header}", exist_ok=True)
            self.Text_log.append(
                f"Default saving directory created : {self.save_header}"
            )
            self.Text_save.setText(
                f"(Default) Save as : {self.save_header}/YYYYMMDD_HHMMSS/denoised.tif\n\nYou can change it using Browse button."
            )
            self.Text_save.setTextColor(QColor(255, 0, 0))
            self.Text_save.append(
                "\n\nWARNING : First 30 and last 30 denoised slices may not accurate result.\nRefer github.com/SUPPORT/issues/X."
            )
            self.Text_save.setTextColor(QColor(0, 0, 0))

            self.savepath_path.setText(
                f"{self.save_header}/YYYYMMDD_HHMMSS/denoised.tif"
            )
            if self.model_path is None:
                self.set_model_path("1")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SUPPORTGUI()
    sys.exit(app.exec_())

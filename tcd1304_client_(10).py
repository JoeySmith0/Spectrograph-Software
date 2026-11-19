#!/usr/bin/env python3
"""
TCD1304 Spectrometer client - Project SunbYte (University of Sheffield)

Layout modelled on the Curious Scientist TCD1304 spectrometer GUI:
  * large filled intensity chart (with "Save chart" button)
  * bottom-left : frame counter, integration time, single/averaging mode,
                  Start / Stop, "Save to file", message box
  * bottom-mid  : 4-point wavelength calibration (Apply / Use calibration parameters)
  * bottom-right: COM port, status, serial message box

Install:  pip install PyQt5 pyqtgraph numpy pyserial
Run:      python tcd1304_client_(10).py       (choose "DEMO" in the COM Port list to test
                                          without hardware)

!! HARDWARE PROTOCOL - edit SerialSpectrometer.acquire() to match your firmware !!
  PC -> MCU : ASCII line "<SH_us>,<ICG_us>,<N_avg>\\n"
  MCU -> PC : 3694 little-endian uint16 samples (7388 bytes)
  Integration time is sent as the SH period; the ICG period is the smallest
  multiple of SH that is longer than the sensor read-out (MIN_READOUT_US).
"""
import sys
import os
import json
import math
import time
import threading
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:
    serial = None
    list_ports = None

N_PIXELS = 3694
DUMMY_START = 32              # first effective pixel (S1) sits at index 32
ACTIVE = 3648                 # effective pixels; the other 46 elements are dummy / shielded
SHOW_DUMMY_AT_START = False   # default state of the "Dummy pixels" button
ADC_BITS = 12
DARK_FRAMES = 10                # frames averaged for a dark-frame capture
CLIP_MARGIN = 2                # counts from an ADC rail that still count as "clipped"
INVERT = True                 # TCD1304 output falls with light -> flip so peaks point up
# Read-out timing (TCD1304DG datasheet): 3694 elements are clocked out per line at
# f_DATA = f_phiM / 4 (typ. 2 MHz master clock -> 0.5 MHz data rate -> 7.388 ms).
# The ICG period can never be shorter than this. SET THIS TO YOUR FIRMWARE'S CLOCK.
MASTER_CLOCK_HZ = 2_000_000
READOUT_US = math.ceil(N_PIXELS * 4 / MASTER_CLOCK_HZ * 1e6)
MIN_READOUT_US = READOUT_US + 100   # + small margin for the ICG/SH pulse timing
INTEGRATION_US = [10, 20, 50, 88, 100, 200, 500, 1000, 2000, 5000, 10000,
                  20000, 50000, 100000, 200000, 500000, 1000000]
CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
DEMO = "DEMO"


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #
def wavelength_to_rgb(wl):
    """Approximate colour of a visible wavelength (nm); grey outside 380-780 nm."""
    if 380 <= wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif 440 <= wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif 490 <= wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif 580 <= wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    elif 645 <= wl <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        return (175, 175, 175)
    if wl < 420:
        f = 0.3 + 0.7 * (wl - 380) / 40
    elif wl > 700:
        f = 0.3 + 0.7 * (780 - wl) / 80
    else:
        f = 1.0
    return tuple(int(255 * (c * f) ** 0.8) for c in (r, g, b))


class SerialSpectrometer:
    def __init__(self, port, baud=115200):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self.ser = serial.Serial(port, baud, timeout=3)
        self.lock = threading.Lock()
        time.sleep(1.5)                       # many boards reset on open

    def acquire(self, sh, icg, avg):
        n = N_PIXELS * 2
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(f"{sh},{icg},{avg}\n".encode())
            buf = self.ser.read(n)
        if len(buf) != n:
            raise IOError(f"Timeout: got {len(buf)} of {n} bytes")
        return np.frombuffer(buf, dtype="<u2").astype(float)

    def send_text(self, text):
        with self.lock:
            self.ser.write((text + "\n").encode())

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


class DemoSpectrometer:
    """Fake CCD (inverted 12-bit output) that mimics the real timing and response:
       * each frame takes one ICG period (x N hardware averages) plus a fixed
         serial-transfer time, so the frame rate follows the integration time
       * line intensity scales with integration time (reference: 100 us) and
         saturates at the ADC limit; a little dark current builds up as well."""
    PEAKS = [(1000, 0.12, 8), (1430, 0.45, 60), (1540, 0.5, 25), (1830, 1.0, 7),
             (2570, 0.1, 6), (2780, 0.38, 9), (3130, 0.12, 10), (3330, 0.04, 12)]
    TRANSFER_S = 0.02             # assumed time to send 7388 bytes to the PC

    def __init__(self):
        self.abort = threading.Event()
        self.covered = False          # demo only: simulates a cap over the sensor

    def set_covered(self, covered):
        self.covered = covered

    def acquire(self, sh, icg, avg):
        # interruptible wait so Stop responds immediately even for long exposures
        if self.abort.wait(icg * avg * 1e-6 + self.TRANSFER_S):
            return None
        x = np.arange(N_PIXELS)
        sig = np.full(N_PIXELS, 500.0 + 0.0005 * sh)           # offset + dark current
        for c, a, w in ([] if self.covered else self.PEAKS):
            sig += a * 3000 * (sh / 100.0) * np.exp(-0.5 * ((x - c) / w) ** 2)
        sig += np.random.normal(0, 8 / math.sqrt(max(avg, 1)), N_PIXELS)
        return np.clip(2 ** ADC_BITS - 1 - sig, 0, 2 ** ADC_BITS - 1)

    def send_text(self, text):
        pass

    def close(self):
        pass


class Worker(QtCore.QThread):
    frame = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, device, params):
        super().__init__()
        self.device, self.params, self._run = device, params, True

    def run(self):
        abort = getattr(self.device, "abort", None)
        if abort is not None:
            abort.clear()
        while self._run:
            try:
                data = self.device.acquire(*self.params)
            except Exception as e:
                self.error.emit(str(e))
                break
            if data is not None and self._run:
                self.frame.emit(data)

    def stop(self):
        self._run = False
        abort = getattr(self.device, "abort", None)
        if abort is not None:
            abort.set()
        self.wait(8000)


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #
class ScaleAxis(pg.AxisItem):
    """Vertical axis that rescales the plot when dragged (manual mode only).
       Drag up = zoom in, drag down = zoom out; the bottom of the axis stays put.
       The mouse wheel over the axis does the same."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manual = False

    def set_manual(self, on):
        self.manual = on
        self.setCursor(QtCore.Qt.SizeVerCursor if on else QtCore.Qt.ArrowCursor)

    def _rescale(self, factor):
        vb = self.linkedView()
        if vb is None:
            return
        lo, hi = vb.viewRange()[1]
        vb.setYRange(lo, lo + max((hi - lo) * factor, 1e-6), padding=0)

    def mouseDragEvent(self, ev):
        if not self.manual or ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        dy = ev.pos().y() - ev.lastPos().y()          # item coords: down is positive
        self._rescale(math.exp(dy * 0.01))

    def wheelEvent(self, ev):
        if not self.manual:
            ev.ignore()
            return
        ev.accept()
        d = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        self._rescale(0.9 ** (d / 120.0))


class ChartWidget(pg.PlotWidget):
    """White plot. Its 'Save chart' / 'Auto-scale Y' buttons live in a toolbar row under the plot
       (so they never cover axis tick labels); the readout and clipping alert overlay the top."""
    def __init__(self):
        super().__init__(background="w", axisItems={"left": ScaleAxis("left")})
        self.save_btn = QtWidgets.QPushButton("Save chart")
        self.save_btn.setFixedSize(92, 26)
        self.auto_chk = QtWidgets.QPushButton("Auto-scale Y: ON")
        self.auto_chk.setCheckable(True)
        self.auto_chk.setChecked(True)
        self.auto_chk.setFixedSize(140, 26)
        
        self.auto_chk.setToolTip("Click to switch automatic vertical scaling on/off.\n"
                                 "When OFF, drag the vertical axis (or scroll over it) to rescale.")
        for b in (self.save_btn, self.auto_chk):
            b.setCursor(QtCore.Qt.PointingHandCursor)
        # clipping alert: small, top-right, ignores the mouse so it never blocks clicks
        self.alert = QtWidgets.QLabel("", self)
        self.alert.setStyleSheet("background: rgba(255,170,0,235); color: black; font-weight: bold;"
                                 " border: 1px solid #a66a00; padding: 3px;")
        self.alert.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.alert.hide()
        self.readout = QtWidgets.QLabel("Click the spectrum to read a pixel", self)
        self.readout.setStyleSheet("background: rgba(255,255,255,200); color: black;"
                                   " border: 1px solid #aaa; padding: 3px;")
        self.readout.move(70, 8)
        self.readout.adjustSize()

    def _place_alert(self):
        a = self.__dict__.get("alert")
        if a is not None:
            a.adjustSize()
            a.move(self.width() - a.width() - 10, 8)

    def set_alert(self, text):
        self.alert.setText(text)
        self.alert.setVisible(bool(text))
        self._place_alert()

    def set_readout(self, text):
        self.readout.setText(text)
        self.readout.adjustSize()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_alert()          # guarded: pyqtgraph calls this during its own __init__


class PortCombo(QtWidgets.QComboBox):
    """Re-scans the serial ports every time the drop-down is opened."""
    def showPopup(self):
        cur = self.currentText()
        self.blockSignals(True)
        self.clear()
        self.addItem("")
        if list_ports:
            self.addItems([p.device for p in list_ports.comports()])
        self.addItem(DEMO)
        i = self.findText(cur)
        self.setCurrentIndex(max(i, 0))
        self.blockSignals(False)
        super().showPopup()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TCD1304 Spectrometer")
        self.resize(1020, 640)
        self.setMinimumSize(900, 560)
        # status bar carries the size grip: drag the bottom-right corner to resize diagonally
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)
        sb.showMessage("Click the spectrum to read a pixel  |  drag the bottom-right corner to resize")
        self.device = None
        self.worker = None
        self.coef = None
        self.frames = 0
        self.outfile = None
        self.last_y = None
        self.sel_px = None            # selected position, in (fractional) pixels
        self.last_base = None         # last frame before dark subtraction
        self.dark = None
        self.dark_us = None
        self.dark_active = False
        self.dark_skip = 0
        self.dark_acc = []
        self.fill_key = None
        self._build_ui()
        self._load_calibration()

    # ---------------------------------------------------------------- UI --- #
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ---- chart
        self.chart = ChartWidget()
        pi = self.chart.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.25)
        pi.setMenuEnabled(False)
        self.chart.setMouseEnabled(False, False)
        for ax in ("left", "bottom"):
            a = pi.getAxis(ax)
            a.setPen(pg.mkPen("k"))
            a.setTextPen(pg.mkPen("k"))
        pi.setLabel("left", "Intensity [a.u.]", color="#000000")
        pi.setLabel("bottom", "Pixels", color="#000000")
        self.curve = pi.plot(pen=pg.mkPen("#1b2a3a", width=1), fillLevel=0,
                             brush=pg.mkBrush(125, 200, 255))
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#d32f2f", width=1.5, style=QtCore.Qt.DashLine))
        self.vline.setZValue(10)
        self.vline.hide()
        pi.addItem(self.vline)
        self.chart.scene().sigMouseClicked.connect(self.on_chart_click)
        pi.setXRange(DUMMY_START, DUMMY_START + ACTIVE - 1, padding=0)
        pi.setYRange(0, 4000, padding=0)
        self.chart.save_btn.clicked.connect(self.save_chart)
        self.chart.auto_chk.toggled.connect(self.autoscale_toggled)
        self.chart.setMinimumHeight(250)
        root.addWidget(self.chart, 1)
        tools = QtWidgets.QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.addWidget(self.chart.save_btn)
        tools.addWidget(self.chart.auto_chk)
        self.btn_dummy = QtWidgets.QPushButton()
        self.btn_dummy.setCheckable(True)
        self.btn_dummy.setChecked(SHOW_DUMMY_AT_START)
        self.btn_dummy.setFixedSize(200, 26)
        self.btn_dummy.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_dummy.setToolTip("Show or hide the 46 dummy / light-shielded elements "
                                  "(pixels 0-31 and 3680-3693).\nPixel numbers do not change.")
        self.btn_dummy.toggled.connect(self.dummy_toggled)
        self.dummy_toggled(SHOW_DUMMY_AT_START)
        tools.addWidget(self.btn_dummy)
        tools.addStretch(1)
        root.addLayout(tools)

        bottom = QtWidgets.QHBoxLayout()
        root.addLayout(bottom)

        # ---- bottom left: run controls
        left = QtWidgets.QGridLayout()
        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(450)
        bottom.addWidget(left_w)
        self.lbl_frames = QtWidgets.QLabel("Number of frames shown: 0")
        left.addWidget(self.lbl_frames, 0, 0, 1, 2)
        left.addWidget(QtWidgets.QLabel("Integration time"), 1, 1)

        self.rb_single = QtWidgets.QRadioButton("Single exposure mode")
        self.rb_avg = QtWidgets.QRadioButton("Averaging mode")
        self.rb_single.setChecked(True)
        self.avg_n = QtWidgets.QSpinBox()
        self.avg_n.setRange(2, 1000)
        self.avg_n.setValue(10)
        self.avg_n.setPrefix("N = ")
        self.avg_n.setMinimumWidth(95)
        avg_row = QtWidgets.QHBoxLayout()
        self.rb_avg.setMinimumWidth(130)
        self.rb_single.setMinimumWidth(170)
        avg_row.addWidget(self.rb_avg)
        avg_row.addWidget(self.avg_n)
        avg_row.addStretch(1)
        radios = QtWidgets.QVBoxLayout()
        radios.addWidget(self.rb_single)
        radios.addLayout(avg_row)
        left.addLayout(radios, 2, 0, 1, 1)

        self.cb_int = QtWidgets.QComboBox()
        for us in INTEGRATION_US:
            self.cb_int.addItem(f"{us} us", us)
        self.cb_int.setCurrentIndex(INTEGRATION_US.index(100))
        left.addWidget(self.cb_int, 2, 1)

        btns = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        for b in (self.btn_start, self.btn_stop):
            b.setFixedSize(80, 40)
            btns.addWidget(b)
        btns.addStretch(1)
        left.addLayout(btns, 3, 0, 1, 2)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        self.chk_save = QtWidgets.QCheckBox("Save to file")
        self.chk_save.toggled.connect(self.toggle_save)
        dark_row = QtWidgets.QHBoxLayout()
        self.btn_dark = QtWidgets.QPushButton("Take dark frame")
        self.btn_dark.setToolTip("Cover the sensor, make sure acquisition is running, then click.\n"
                                 f"The next {DARK_FRAMES} frames are averaged and stored as the dark frame.")
        self.btn_dark.clicked.connect(self.take_dark)
        self.chk_dark = QtWidgets.QCheckBox("Subtract dark")
        self.chk_dark.setEnabled(False)
        self.chk_dark.toggled.connect(self.redisplay)
        self.lbl_dark = QtWidgets.QLabel("")
        dark_row.addWidget(self.btn_dark)
        dark_row.addWidget(self.chk_dark)
        dark_row.addWidget(self.lbl_dark)
        dark_row.addStretch(1)
        left.addLayout(dark_row, 4, 0, 1, 2)
        self.cb_int.currentIndexChanged.connect(self.update_dark_label)
        left.addWidget(self.chk_save, 5, 0, 1, 2)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(70)
        left.addWidget(self.log_box, 6, 0, 1, 2)

        # ---- bottom middle: calibration
        cal = QtWidgets.QGroupBox("Calibration parameters")
        g = QtWidgets.QGridLayout(cal)
        self.wl_edits, self.px_edits = [], []
        for i in range(4):
            w, p = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
            w.setFixedWidth(90)
            p.setFixedWidth(90)
            self.wl_edits.append(w)
            self.px_edits.append(p)
            g.addWidget(QtWidgets.QLabel(f"Wavelength {i + 1} (nm)"), i, 0)
            g.addWidget(w, i, 1)
            g.addWidget(QtWidgets.QLabel(f"Pixel-peak {i + 1}"), i, 2)
            g.addWidget(p, i, 3)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply_calibration)
        g.addWidget(self.btn_apply, 0, 4, 2, 1)
        self.lbl_fit = QtWidgets.QLabel("Fitting parameters:")
        self.lbl_fit.setWordWrap(True)
        g.addWidget(self.lbl_fit, 4, 0, 1, 3)
        self.chk_cal = QtWidgets.QCheckBox("Use calibration parameters")
        self.chk_cal.setEnabled(False)
        self.chk_cal.toggled.connect(self.update_axis)
        g.addWidget(self.chk_cal, 4, 3, 1, 2)
        bottom.addWidget(cal, 1)

        # ---- bottom right: COM port
        right = QtWidgets.QVBoxLayout()
        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)
        right_w.setFixedWidth(200)
        bottom.addWidget(right_w)
        right.addWidget(QtWidgets.QLabel("COM Port"))
        self.cb_port = PortCombo()
        self.cb_port.addItem("")
        if list_ports:
            self.cb_port.addItems([p.device for p in list_ports.comports()])
        self.cb_port.addItem(DEMO)
        self.cb_port.currentIndexChanged.connect(self.port_changed)
        right.addWidget(self.cb_port)
        self.lbl_status = QtWidgets.QLabel("Status: Disconnected")
        right.addWidget(self.lbl_status)
        right.addStretch(1)
        right.addWidget(QtWidgets.QLabel("Serial message"))
        row = QtWidgets.QHBoxLayout()
        self.ed_msg = QtWidgets.QLineEdit()
        self.btn_send = QtWidgets.QPushButton("Send")
        self.btn_send.setFixedWidth(60)
        self.ed_msg.returnPressed.connect(self.send_message)
        self.btn_send.clicked.connect(self.send_message)
        row.addWidget(self.ed_msg)
        row.addWidget(self.btn_send)
        right.addLayout(row)

        self._set_connected(False)

    # ---------------------------------------------------------- helpers --- #
    def log(self, text):
        self.log_box.appendPlainText(text)

    def _set_connected(self, on):
        for w in (self.rb_single, self.rb_avg, self.avg_n, self.cb_int,
                  self.btn_start, self.btn_send, self.ed_msg, self.btn_dark):
            w.setEnabled(on)
        self.btn_stop.setEnabled(False)

    def _set_running(self, running):
        for w in (self.rb_single, self.rb_avg, self.avg_n, self.cb_int, self.btn_start):
            w.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    # -------------------------------------------------------- connection --- #
    def port_changed(self):
        self.stop()
        if self.device:
            self.device.close()
            self.device = None
        port = self.cb_port.currentText()
        if not port:
            self.lbl_status.setText("Status: Disconnected")
            self._set_connected(False)
            return
        try:
            self.device = DemoSpectrometer() if port == DEMO else SerialSpectrometer(port)
        except Exception as e:
            self.lbl_status.setText("Status: Error")
            self.log(f"Could not open {port}: {e}")
            self._set_connected(False)
            return
        self.lbl_status.setText("Status: Connected")
        self.log(f"Connected to {port}.")
        self._set_connected(True)

    def send_message(self):
        txt = self.ed_msg.text()
        if self.device and txt:
            try:
                self.device.send_text(txt)
                self.log(f"> {txt}")
                self.ed_msg.clear()
            except Exception as e:
                self.log(f"Send failed: {e}")

    # ------------------------------------------------------- acquisition --- #
    def start(self):
        if not self.device:
            return
        us = self.cb_int.currentData()
        icg = us * math.ceil(MIN_READOUT_US / us)
        avg = self.avg_n.value() if self.rb_avg.isChecked() else 1
        self.frames = 0
        self.lbl_frames.setText("Number of frames shown: 0")
        self.worker = Worker(self.device, (us, icg, avg))
        self.worker.frame.connect(self.on_frame)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        self._set_running(True)

    def stop(self):
        self._cancel_dark()
        if self.worker:
            self.worker.stop()
            self.worker = None
            self._set_running(False)

    def on_error(self, msg):
        self.log(f"Error: {msg}")
        self.lbl_status.setText("Status: Error")
        self.stop()

    def check_clipping(self, raw):
        """Flag pixels sitting on an ADC rail: their true intensity is unknown."""
        full = 2 ** ADC_BITS - 1
        r = raw[self.visible_slice()]
        at_max = int(np.count_nonzero(r >= full - CLIP_MARGIN))
        at_min = int(np.count_nonzero(r <= CLIP_MARGIN))
        bright, dark = (at_min, at_max) if INVERT else (at_max, at_min)
        lines = []
        if bright:
            lines.append(f"⚠ CLIPPING: {bright} px saturated – reduce integration time")
        if dark:
            lines.append(f"⚠ {dark} px below the ADC floor")
        self.chart.set_alert("\n".join(lines))

    def on_frame(self, raw):
        self.check_clipping(raw)
        base = (2 ** ADC_BITS - 1 - raw) if INVERT else raw
        self.last_base = base
        if self.dark_active:
            self._dark_step(base)
        y = self.processed(base)
        self.frames += 1
        self.lbl_frames.setText(f"Number of frames shown: {self.frames}")
        self.draw(y)
        if self.outfile:
            line = f"{time.time():.3f},{self.cb_int.currentData()}," + ",".join(f"{v:.1f}" for v in y)
            self.outfile.write(line + "\n")

    # -------------------------------------------------------------- plot --- #
    # ---------------------------------------------------------- dark frame --- #
    def processed(self, base):
        if self.chk_dark.isChecked() and self.dark is not None:
            return base - self.dark
        return base

    def redisplay(self, *_):
        if self.last_base is not None:
            self.draw(self.processed(self.last_base))

    def take_dark(self):
        if self.worker is None:
            self.log("Start acquisition first, then take a dark frame.")
            return
        if hasattr(self.device, "set_covered"):
            self.device.set_covered(True)           # demo device: simulate covering the sensor
        self.dark_acc = []
        self.dark_skip = 1                          # discard a frame already in flight
        self.dark_active = True
        self.btn_dark.setEnabled(False)
        self.log(f"Taking dark frame: keep the sensor covered ({DARK_FRAMES} frames)...")

    def _dark_step(self, base):
        if self.dark_skip > 0:
            self.dark_skip -= 1
            return
        self.dark_acc.append(base.copy())
        if len(self.dark_acc) >= DARK_FRAMES:
            self.dark = np.mean(self.dark_acc, axis=0)
            self.dark_us = self.cb_int.currentData()
            self._end_dark()
            self.chk_dark.setEnabled(True)
            self.chk_dark.setChecked(True)          # also redraws
            self.update_dark_label()
            self.log(f"Dark frame stored (mean level {self.dark.mean():.0f} counts).")

    def _end_dark(self):
        self.dark_active = False
        if hasattr(self.device, "set_covered"):
            self.device.set_covered(False)
        self.btn_dark.setEnabled(self.device is not None)

    def _cancel_dark(self):
        if self.dark_active:
            self._end_dark()
            self.log("Dark frame cancelled.")

    def update_dark_label(self, *_):
        if self.dark is None:
            self.lbl_dark.setText("")
            return
        same = self.cb_int.currentData() == self.dark_us
        self.lbl_dark.setText(f"({self.dark_us} us)" + ("" if same else " ⚠"))
        self.lbl_dark.setStyleSheet("" if same else "color: #b36b00; font-weight: bold;")
        self.lbl_dark.setToolTip("" if same else
                                 "Dark level grows with integration time: this dark frame was "
                                 "taken at a different setting. Take a new one.")

    # ------------------------------------------------------ colour fill ---- #
    def update_fill(self, x):
        cal = self.chk_cal.isChecked() and self.coef is not None
        key = (round(float(x.min()), 3), round(float(x.max()), 3)) if cal else None
        if key == self.fill_key:
            return
        self.fill_key = key
        if key is None or key[1] - key[0] < 1e-6:
            brush = pg.mkBrush(125, 200, 255)
        else:
            lo, hi = key
            grad = QtGui.QLinearGradient(lo, 0, hi, 0)      # data coordinates
            for t in np.linspace(0.0, 1.0, 150):
                grad.setColorAt(float(t), QtGui.QColor(*wavelength_to_rgb(lo + t * (hi - lo))))
            brush = QtGui.QBrush(grad)
        self.curve.setFillBrush(brush)

    def x_values(self):
        px = np.arange(N_PIXELS, dtype=float)
        if self.chk_cal.isChecked() and self.coef is not None:
            return np.polyval(self.coef, px)
        return px

    def draw(self, y):
        self.last_y = y
        sl = self.visible_slice()
        x = self.x_values()[sl]
        yv = y[sl]
        self.update_fill(x)
        self.curve.setData(x, yv)
        self.update_selection()
        if self.chart.auto_chk.isChecked():
            top = max(float(np.max(yv)) * 1.08, 10.0)
            bot = min(float(np.min(yv)), 0.0)
            self.chart.getPlotItem().setYRange(bot, top, padding=0)

    def visible_slice(self):
        if self.btn_dummy.isChecked():
            return slice(0, N_PIXELS)
        return slice(DUMMY_START, DUMMY_START + ACTIVE)

    def dummy_toggled(self, shown):
        self.btn_dummy.setText("Dummy pixels: SHOWN" if shown else "Dummy pixels: HIDDEN")
        if hasattr(self, "chart") and hasattr(self, "chk_cal"):
            self.update_axis()

    def autoscale_toggled(self, auto):
        self.chart.auto_chk.setText("Auto-scale Y: ON" if auto else "Auto-scale Y: OFF")
        self.chart.getPlotItem().getAxis("left").set_manual(not auto)
        if auto:
            if self.last_y is not None:
                self.draw(self.last_y)
            self.log("Y axis: automatic scaling.")
        else:
            self.log("Y axis: fixed. Drag the vertical axis (or scroll over it) to rescale.")

    def update_axis(self):
        pi = self.chart.getPlotItem()
        cal = self.chk_cal.isChecked() and self.coef is not None
        pi.setLabel("bottom", "Wavelength (nm)" if cal else "Pixels", color="#000000")
        x = self.x_values()[self.visible_slice()]
        pi.setXRange(float(x.min()), float(x.max()), padding=0)
        if self.last_y is not None:
            self.draw(self.last_y)
        else:
            self.update_selection()

    # ------------------------------------------------- click to read pixel --- #
    def px_to_x(self, px):
        return float(np.interp(px, np.arange(N_PIXELS), self.x_values()))

    def x_to_px(self, xv):
        """Axis coordinate -> fractional pixel (interpolated, so not limited to integers)."""
        if not (self.chk_cal.isChecked() and self.coef is not None):
            return float(np.clip(xv, 0, N_PIXELS - 1))
        x = self.x_values()
        order = np.argsort(x)
        return float(np.interp(xv, x[order], np.arange(N_PIXELS)[order]))

    @staticmethod
    def refine_peak(y, i):
        """Sub-pixel peak position from a parabola through the 3 points around index i."""
        if 0 < i < len(y) - 1:
            a, b, c = y[i - 1], y[i], y[i + 1]
            den = a - 2 * b + c
            if den < 0:
                return i + 0.5 * (a - c) / den
        return float(i)

    def on_chart_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.chart.getPlotItem().vb
        pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        self.sel_px = self.x_to_px(vb.mapSceneToView(pos).x())
        self.chart.setFocus()                       # so Left/Right keys can nudge
        self.update_selection(announce=True)

    def update_selection(self, announce=False):
        if self.sel_px is None:
            return
        px = self.sel_px
        xs = self.px_to_x(px)
        cal = self.chk_cal.isChecked() and self.coef is not None
        self.vline.setValue(xs)
        self.vline.show()
        lines = [f"Pixel: {px:.2f}" + (f"    Wavelength: {xs:.2f} nm" if cal else "")]
        if self.last_y is not None:
            idx = np.arange(N_PIXELS)
            c = int(round(px))
            vis = self.visible_slice()
            lo, hi = max(vis.start, c - 10), min(vis.stop, c + 11)
            txt = f"Intensity: {np.interp(px, idx, self.last_y):.0f}"
            if hi > lo:
                pk = self.refine_peak(self.last_y, lo + int(np.argmax(self.last_y[lo:hi])))
                txt += (f"    Local peak (±10 px): pixel {pk:.2f}"
                        + (f" ({self.px_to_x(pk):.2f} nm)" if cal else ""))
            lines.append(txt)
        lines.append("(←/→ nudge 0.1 px, Shift+←/→ nudge 1 px)")
        self.chart.set_readout("\n".join(lines))
        if announce:
            QtWidgets.QApplication.clipboard().setText(f"{px:.2f}")
            self.log(f"Selected pixel {px:.2f} (copied to clipboard)")

    def keyPressEvent(self, e):
        if self.sel_px is not None and e.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            step = 1.0 if e.modifiers() & QtCore.Qt.ShiftModifier else 0.1
            if e.key() == QtCore.Qt.Key_Left:
                step = -step
            self.sel_px = float(np.clip(round(self.sel_px + step, 3), 0, N_PIXELS - 1))
            self.update_selection(announce=True)
        else:
            super().keyPressEvent(e)

    def save_chart(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save chart", time.strftime("spectrum_%Y%m%d_%H%M%S.png"), "PNG (*.png)")
        if fn:
            from pyqtgraph.exporters import ImageExporter
            ImageExporter(self.chart.getPlotItem()).export(fn)
            self.log(f"Chart saved: {os.path.basename(fn)}")

    # ------------------------------------------------------- save to file -- #
    def toggle_save(self, on):
        if on:
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save frames to", time.strftime("frames_%Y%m%d_%H%M%S.csv"), "CSV (*.csv)")
            if not fn:
                self.chk_save.setChecked(False)
                return
            self.outfile = open(fn, "a", buffering=1)
            self.outfile.write("# unix_time,integration_us,pixel0..pixel3693\n")
            self.log(f"Saving frames to {os.path.basename(fn)}")
        elif self.outfile:
            self.outfile.close()
            self.outfile = None
            self.log("Stopped saving.")

    # ------------------------------------------------------- calibration --- #
    def apply_calibration(self):
        pts = []
        for w, p in zip(self.wl_edits, self.px_edits):
            try:
                pts.append((float(p.text()), float(w.text())))
            except ValueError:
                pass
        if len(pts) < 2 or len({p for p, _ in pts}) < len(pts):
            self.log("Need at least 2 valid wavelength/pixel pairs (distinct pixels).")
            return
        px, wl = np.array(pts).T
        deg = min(2, len(pts) - 1)                    # linear with 2 pts, else quadratic
        self.coef = np.polyfit(px, wl, deg)
        names = "abc"[2 - deg:]          # y = a x^2 + b x + c (or b x + c)
        self.lbl_fit.setText("Fitting parameters: " + "  ".join(
            f"{n}={c:.6g}" for n, c in zip(names, self.coef)))
        self.chk_cal.setEnabled(True)
        try:
            with open(CAL_FILE, "w") as f:
                json.dump({"wavelength": [e.text() for e in self.wl_edits],
                           "pixel": [e.text() for e in self.px_edits]}, f)
            self.log("Calibration applied and saved.")
        except OSError as e:
            self.log(f"Calibration applied (could not save file: {e})")
        self.update_axis()

    def _load_calibration(self):
        try:
            with open(CAL_FILE) as f:
                d = json.load(f)
            for e, t in zip(self.wl_edits, d["wavelength"]):
                e.setText(t)
            for e, t in zip(self.px_edits, d["pixel"]):
                e.setText(t)
            self.apply_calibration()
            self.log("Calibration file is loaded.")
        except (OSError, KeyError, ValueError):
            self.log("No calibration file found.")

    def closeEvent(self, e):
        self.stop()
        if self.device:
            self.device.close()
        if self.outfile:
            self.outfile.close()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#f0f0f0"))
    app.setPalette(pal)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

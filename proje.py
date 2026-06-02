# =========================================================
# AGRIVISION AI - ENTERPRISE DASHBOARD EDITION (V20.3 BUGFIX)
# =========================================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys

# GPU Devre Dışı Bırakma
sys.argv.append("--disable-gpu")
sys.argv.append("--software-rendering-fps=60")

import requests
import base64
import cv2
import json
import h5py
import numpy as np
import tensorflow as tf
import ee
from io import BytesIO
from rasterio.io import MemoryFile
import time
tf.config.set_visible_devices([], 'GPU')

import rasterio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QFileDialog, QComboBox, QLabel, 
                             QGroupBox, QSizePolicy, QFrame, QScrollArea, QStackedWidget,
                             QDateEdit, QSlider, QGridLayout, QGraphicsDropShadowEffect,
                             QProgressBar, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QTextBrowser, QLineEdit, QDialog, QAbstractItemView, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter, QGraphicsOpacityEffect)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QIcon, QColor, QBrush, QDrag
from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QTimer, QDate, QAbstractAnimation, pyqtSignal, QMimeData

# --- PIE CHART İÇİN KÜTÜPHANELER ---
try:
    from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice
except ImportError:
    print("HATA: Lütfen terminalden 'pip install PyQtChart' komutunu çalıştırın.")
    sys.exit(1)

import random
import sqlite3
import hashlib
from datetime import datetime

# =========================================================
# CUSTOM SÜRÜKLENEBİLİR BUTON (DRAG & DROP)
# =========================================================
class DraggableLayerButton(QPushButton):
    def __init__(self, text, code, parent=None):
        super().__init__(text, parent)
        self.code = code
        self.setCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.code)
            drag.setMimeData(mime)
            
            pixmap = self.grab()
            drag.setPixmap(pixmap)
            drag.setHotSpot(event.pos())
            
            drag.exec_(Qt.MoveAction)
        super().mouseMoveEvent(event)

# =========================================================
# YENİ NESİL TIKLANABİLİR VE YAKINLAŞTIRILABİLİR HARİTA
# =========================================================
class AdvancedMapViewer(QGraphicsView):
    clicked_pos = pyqtSignal(int, int)
    view_changed = pyqtSignal()
    layer_dropped = pyqtSignal(str, object) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True) 

        self.horizontalScrollBar().valueChanged.connect(self.emit_view_changed)
        self.verticalScrollBar().valueChanged.connect(self.emit_view_changed)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        layer_code = event.mimeData().text()
        self.layer_dropped.emit(layer_code, self)
        event.acceptProposedAction()

    def set_image(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())

    def emit_view_changed(self):
        self.view_changed.emit()

    def zoom_in(self):
        self.scale(1.2, 1.2)
        self.view_changed.emit()

    def zoom_out(self):
        self.scale(1.0 / 1.2, 1.0 / 1.2)
        self.view_changed.emit()

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        if event.angleDelta().y() > 0: zoom_factor = zoom_in_factor
        else: zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)
        self.view_changed.emit()
        event.accept() 

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.clicked_pos.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mousePressEvent(event)

# =========================================================
# İNDİRME SEÇİM PENCERESİ
# =========================================================
class DownloadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Katman İndir")
        self.setFixedSize(350, 180)
        self.setStyleSheet("background-color: white; border-radius: 12px;")
        
        layout = QVBoxLayout(self)
        
        lbl = QLabel("İndirmek istediğiniz haritayı seçin:")
        lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #1e293b;")
        layout.addWidget(lbl)
        
        self.combo = QComboBox()
        self.combo.addItems(["🤖 AI Segmentasyon Maskesi", "🌍 Orijinal Uydu (RGB)", "🔴 Kızılötesi (FC)", "🌿 NDVI (Bitki Sağlığı)"])
        self.combo.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; padding: 10px; border-radius: 8px; font-weight: bold;")
        layout.addWidget(self.combo)
        
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("İptal")
        self.btn_cancel.setStyleSheet("background-color: #e2e8f0; padding: 10px; border-radius: 8px; font-weight: bold;")
        self.btn_cancel.clicked.connect(self.reject)
        
        self.btn_ok = QPushButton("İndir")
        self.btn_ok.setStyleSheet("background-color: #10b981; color: white; padding: 10px; border-radius: 8px; font-weight: bold;")
        self.btn_ok.clicked.connect(self.accept)
        
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def get_selected_layer(self):
        idx = self.combo.currentIndex()
        codes = ["ai", "rgb", "fc", "ndvi"]
        return codes[idx]

# =========================================================
# H5 FİX 
# =========================================================
def fix_keras3_h5(file_path):
    try:
        with h5py.File(file_path, 'r+') as f:
            if 'model_config' not in f.attrs: return
            config = json.loads(f.attrs['model_config'])
            modified = False
            if 'config' in config and 'layers' in config['config']:
                for layer in config['config']['layers']:
                    l_config = layer.get('config', {})
                    if isinstance(l_config.get('dtype'), dict):
                        if l_config['dtype'].get('class_name') == 'DTypePolicy':
                            l_config['dtype'] = 'float32'
                            modified = True
                    if layer.get('class_name') == 'BatchNormalization':
                        l_config.pop('synchronized', None)
                if modified:
                    f.attrs['model_config'] = json.dumps(config).encode('utf-8')
    except Exception as e:
        pass

# =========================================================
# MAIN APP
# =========================================================
class AgriVisionAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AgriVision AI - Enterprise Dashboard")
        self.setGeometry(50, 30, 1800, 950)

        self.raster_data = None
        self.rgb_base = None
        self.fc_base = None    
        self.ndvi_base = None  
        self.mask_overlay = None
        self.pred_map = None
        self.gt_mask = None
        
        self.left_layer_code = "ai"
        self.right_layer_code = "rgb"

        self.models_dict = {
            "AgriVision Pro U-Net (v2.0)": "unet_rekor_model_V2.h5",
            "Attention U-Net Advanced (v3.1)": "attention_unet_model_V3.h5",
            "U-Net PlusPlus Deep (v7.0)": "unet_plus_plus_model_V7.h5",
            "U2-Net Legacy Edition (v6.0)": "u2net_eski_veri_model_ikinci_versiyon_V6.h5"
        }

        self.classes = [
            ("Mısır", "#f0c419"), ("Fındık", "#c57b39"),
            ("Pirinç", "#84cc16"), ("Kavak", "#5dbb63"),  # <-- RENGİ #84cc16 YAPTIK
            ("Su Geçirimsiz", "#b5b5b5"), ("Diğer Vejetasyon", "#9dcc4c"),
            ("Su", "#4b86f0")
        ]

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: #f6f8fb; }")
        
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background: #f6f8fb;")
        self.central_widget.setMinimumHeight(1150) 
        self.scroll_area.setWidget(self.central_widget)
        self.setCentralWidget(self.scroll_area)

        self.setup_ui()
        self.apply_style()
        self.setup_notification()

    def add_shadow(self, widget, radius=25, offset=5, opacity=15):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(radius)
        shadow.setOffset(0, offset)
        shadow.setColor(QColor(0, 0, 0, opacity))
        widget.setGraphicsEffect(shadow)

    def setup_notification(self):
        self.toast = QFrame(self)
        self.toast.setObjectName("toastFrame")
        self.toast.hide()
        toast_layout = QHBoxLayout(self.toast)
        toast_layout.setContentsMargins(30, 20, 20, 20)
        toast_layout.setSpacing(20)
        
        self.toast_lbl = QLabel("")
        self.toast_lbl.setObjectName("toastLabel")
        
        self.toast_btn = QPushButton("✕")
        self.toast_btn.setObjectName("toastCloseBtn")
        self.toast_btn.setFixedSize(36, 36)
        self.toast_btn.setCursor(Qt.PointingHandCursor)
        self.toast_btn.clicked.connect(self.hide_toast)
        
        toast_layout.addWidget(self.toast_lbl)
        toast_layout.addWidget(self.toast_btn)
        
        self.toast_effect = QGraphicsOpacityEffect(self.toast)
        self.toast.setGraphicsEffect(self.toast_effect)
        self.toast_animation = QPropertyAnimation(self.toast_effect, b"opacity")
        self.toast_timer = QTimer(self)
        self.toast_timer.timeout.connect(self.hide_toast)

    def show_toast(self, message):
        self.toast_lbl.setText(message)
        self.toast.adjustSize()
        x = self.width() - self.toast.width() - 30
        y = self.height() - self.toast.height() - 30
        self.toast.move(x, y)
        self.toast.show()
        self.toast_animation.stop()
        self.toast_effect.setOpacity(1.0)
        self.toast_timer.start(7000) 

    def hide_toast(self):
        self.toast_timer.stop()
        self.toast_animation.setDuration(600)
        self.toast_animation.setStartValue(self.toast_effect.opacity())
        self.toast_animation.setEndValue(0.0)
        self.toast_animation.start()
        QTimer.singleShot(600, self.toast.hide)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'toast') and self.toast.isVisible():
            x = self.width() - self.toast.width() - 30
            y = self.height() - self.toast.height() - 30
            self.toast.move(x, y)

    def create_card_container(self, step_text, title_text):
        card = QFrame()
        card.setStyleSheet("QFrame { background: white; border-radius: 16px; border: 1px solid #eef2f7; }")
        self.add_shadow(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        
        if step_text:
            step_lbl = QLabel(step_text)
            step_lbl.setStyleSheet("color: #64748b; font-weight: 900; font-size: 11px; letter-spacing: 1px; border: none;")
            layout.addWidget(step_lbl)
            
        if title_text:
            title_lbl = QLabel(title_text)
            title_lbl.setStyleSheet("color: #1e293b; font-weight: 800; font-size: 16px; margin-bottom: 5px; border: none;")
            layout.addWidget(title_lbl)
        
        content_layout = QVBoxLayout()
        content_layout.setSpacing(10)
        layout.addLayout(content_layout)
        return card, content_layout

    def setup_ui(self):
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(30, 20, 30, 40)
        main_layout.setSpacing(25)

        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 10)
        self.main_title = QLabel("A G R I V I S I O N   A I")
        self.main_title.setAlignment(Qt.AlignCenter)
        self.main_title.setStyleSheet("font-size: 54px; font-weight: 1000; color: #1e293b; letter-spacing: 2px;")
        
        self.sub_title = QLabel("S E M A N T I C   S E G M E N T A T I O N   P A N E L")
        self.sub_title.setAlignment(Qt.AlignCenter)
        self.sub_title.setStyleSheet("font-size: 15px; color: #64748b; letter-spacing: 4px;")
        
        header_layout.addWidget(self.main_title)
        header_layout.addWidget(self.sub_title)
        main_layout.addLayout(header_layout)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(30)

        # ================= SOL PANEL =================
        left_panel = QWidget()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(0, 0, 0, 0)

        card1, cl1 = self.create_card_container("STEP 1", "Veri Yükleme")
        self.btn_tif = QPushButton("📁  10 Bantlı TIF Yükle")
        self.btn_tif.setCursor(Qt.PointingHandCursor)
        self.btn_tif.setStyleSheet("background-color: #10b981; color: black; font-weight: 900; padding: 14px; border-radius: 10px; font-size: 14px;")
        self.btn_tif.clicked.connect(self.load_tif)
        
        self.btn_gt = QPushButton("📝  Manuel Referans Mask Yükle")
        self.btn_gt.setCursor(Qt.PointingHandCursor)
        self.btn_gt.setStyleSheet("background-color: #f59e0b; color: black; font-weight: 900; padding: 14px; border-radius: 10px; font-size: 14px;")
        self.btn_gt.clicked.connect(self.load_gt)
        cl1.addWidget(self.btn_tif)
        cl1.addWidget(self.btn_gt)
        left_layout.addWidget(card1)

        card2, cl2 = self.create_card_container("STEP 2", "Model Seç")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.models_dict.keys())
        self.model_combo.setStyleSheet("background-color: #f8fafc; border: 2px solid #e2e8f0; border-radius: 10px; padding: 12px; font-weight: bold; color: #1e293b;")
        cl2.addWidget(self.model_combo)
        left_layout.addWidget(card2)

        card3, cl3 = self.create_card_container("STEP 3", "Analizi Başlat")
        self.btn_run = QPushButton("🚀  MODELİ ÇALIŞTIR")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setStyleSheet("background-color: #3b82f6; color: white; font-weight: 900; padding: 16px; font-size: 16px; border-radius: 10px;")
        self.btn_run.clicked.connect(self.run_model)
        cl3.addWidget(self.btn_run)
        left_layout.addWidget(card3)

        card_status, cl_status = self.create_card_container("", "Sistem Durumu")
        self.status_label = QLabel("💤 Sistem Hazır. Görüntü bekleniyor...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 13px; color: #64748b; font-weight: 800;")
        
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(18)
        self.progress.setStyleSheet("""
            QProgressBar { background-color: #f1f5f9; border-radius: 9px; border: none; }
            QProgressBar::chunk { background-color: #3b82f6; border-radius: 9px; }
        """)
        cl_status.addWidget(self.status_label)
        cl_status.addWidget(self.progress)
        
        self.btn_export = QPushButton("💾 Haritayı İndir (TIF)")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setStyleSheet("background-color: #0f172a; color: white; font-weight: 900; padding: 12px; font-size: 13px; border-radius: 10px; margin-top: 10px;")
        self.btn_export.clicked.connect(self.prompt_export_layer)
        cl_status.addWidget(self.btn_export)

        left_layout.addWidget(card_status)
        left_layout.addStretch() 
        body_layout.addWidget(left_panel)

        # ================= SAĞ PANEL =================
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(0, 0, 0, 0)

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        self.live_stats = {}
        
        stat_items = [
            ("Toplam Alan", "🌍"), ("Bulunan Sınıf", "🔍"), 
            ("Model", "🧠"), ("İşlem Süresi", "⏱️")
        ]
        
        for stat_name, icon in stat_items:
            stat_card = QFrame()
            stat_card.setFixedHeight(90) 
            stat_card.setStyleSheet("background: white; border-radius: 16px; border: 1px solid #eef2f7;")
            self.add_shadow(stat_card, radius=15, offset=3, opacity=8)
            stat_card_layout = QHBoxLayout(stat_card)
            stat_card_layout.setContentsMargins(20, 15, 20, 15)
            stat_card_layout.setSpacing(15)
            
            icon_lbl = QLabel(icon)
            icon_lbl.setStyleSheet("font-size: 28px; background: transparent; border: none;")
            
            text_layout = QVBoxLayout()
            text_layout.setAlignment(Qt.AlignVCenter)
            title_lbl = QLabel(stat_name)
            title_lbl.setStyleSheet("color: #64748b; font-size: 12px; font-weight: bold; background: transparent; border: none;")
            val_lbl = QLabel("-")
            val_lbl.setStyleSheet("color: #0f172a; font-size: 18px; font-weight: 900; background: transparent; border: none;")
            
            text_layout.addWidget(title_lbl)
            text_layout.addWidget(val_lbl)
            stat_card_layout.addWidget(icon_lbl)
            stat_card_layout.addLayout(text_layout)
            stat_card_layout.addStretch()
            stats_layout.addWidget(stat_card)
            self.live_stats[stat_name] = val_lbl

        right_layout.addLayout(stats_layout)

        # --- HARİTA BÖLÜMÜ (SPLITTER, ZOOM, DRAG-DROP) ---
        map_card, map_layout = self.create_card_container("", "Harita ve Katman Görünümü")
        map_internal_layout = QHBoxLayout()
        map_internal_layout.setSpacing(20)
        
        self.map_container = QWidget()
        mc_layout = QVBoxLayout(self.map_container)
        mc_layout.setContentsMargins(0,0,0,0)
        
        toolbar = QFrame()
        toolbar.setStyleSheet("background: #f1f5f9; border-radius: 8px; padding: 5px;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(5, 5, 5, 5)
        
        self.btn_split = QPushButton("◧")
        self.btn_split.setToolTip("İkili Ekran Karşılaştırma Modu")
        self.btn_split.setCheckable(True)
        self.btn_split.setFixedSize(36, 36)
        self.btn_split.setStyleSheet("QPushButton { background-color: #cbd5e1; border-radius: 6px; font-size: 18px; } QPushButton:checked { background-color: #3b82f6; color: white; }")
        self.btn_split.clicked.connect(self.toggle_split_screen)
        
        self.btn_zoom_in = QPushButton("🔍+")
        self.btn_zoom_in.setFixedSize(36, 36)
        self.btn_zoom_in.setStyleSheet("background-color: white; border: 1px solid #cbd5e1; border-radius: 6px; font-weight:bold;")
        self.btn_zoom_in.clicked.connect(self.zoom_in_maps)
        
        self.btn_zoom_out = QPushButton("🔍-")
        self.btn_zoom_out.setFixedSize(36, 36)
        self.btn_zoom_out.setStyleSheet("background-color: white; border: 1px solid #cbd5e1; border-radius: 6px; font-weight:bold;")
        self.btn_zoom_out.clicked.connect(self.zoom_out_maps)
        
        lbl_info = QLabel(" 💡 Katmanları haritaların üzerine sürükleyip bırakabilirsiniz.")
        lbl_info.setStyleSheet("color: #64748b; font-size: 11px; font-weight: bold; background:transparent;")
        
        tb_layout.addWidget(self.btn_split)
        tb_layout.addWidget(self.btn_zoom_in)
        tb_layout.addWidget(self.btn_zoom_out)
        tb_layout.addWidget(lbl_info)
        tb_layout.addStretch()
        
        mc_layout.addWidget(toolbar)

        self.map_splitter = QSplitter(Qt.Horizontal)
        self.map_splitter.setStyleSheet("QSplitter::handle { background-color: #cbd5e1; width: 4px; }")
        
        self.image_view = AdvancedMapViewer()
        self.image_view.setMinimumHeight(400) 
        self.image_view.layer_dropped.connect(self.on_layer_dropped)
        
        self.split_image_view = AdvancedMapViewer()
        self.split_image_view.setMinimumHeight(400)
        self.split_image_view.hide()
        self.split_image_view.layer_dropped.connect(self.on_layer_dropped)
        
        self._syncing = False
        def sync_maps(source, target):
            if self._syncing or not target.isVisible(): return
            self._syncing = True
            
            # ZIPLAMA HATASINI KESİN ÇÖZEN SİHİRLİ SATIR (Scrollbar yerine Merkeze Odaklanma)
            target.setTransform(source.transform())
            center_point = source.mapToScene(source.viewport().rect().center())
            target.centerOn(center_point)
            
            self._syncing = False

        self.image_view.view_changed.connect(lambda: sync_maps(self.image_view, self.split_image_view))
        self.split_image_view.view_changed.connect(lambda: sync_maps(self.split_image_view, self.image_view))

        self.map_splitter.addWidget(self.image_view)
        self.map_splitter.addWidget(self.split_image_view)
        mc_layout.addWidget(self.map_splitter, stretch=1)
        
        map_internal_layout.addWidget(self.map_container, stretch=1)

        map_controls_widget = QWidget()
        map_controls_widget.setMinimumWidth(260) 
        map_controls_widget.setMaximumWidth(300)
        map_controls_layout = QVBoxLayout(map_controls_widget)
        map_controls_layout.setContentsMargins(0, 0, 0, 0)
        map_controls_layout.setAlignment(Qt.AlignTop)
        map_controls_layout.setSpacing(15)
        
        layer_card = QFrame()
        layer_card.setStyleSheet("background: #f8fafc; border-radius: 12px; border: 1px solid #eef2f7;")
        layer_vbox = QVBoxLayout(layer_card)
        layer_vbox.setContentsMargins(15, 15, 15, 15)
        lbl_k = QLabel("KATMANLAR (Tıkla veya Sürükle)")
        lbl_k.setStyleSheet("color: #64748b; font-weight: bold; font-size: 11px; margin-bottom: 5px;")
        layer_vbox.addWidget(lbl_k)
        
        self.layer_btns = []
        layers_info = [
            ("🌍 Doğal Renkli (RGB)", "rgb"),
            ("🔴 Kızılötesi (FC)", "fc"),
            ("🌿 NDVI İndeksi", "ndvi"),
            ("🤖 AI Segmentasyon", "ai")
        ]
        
        for text, code in layers_info:
            btn = DraggableLayerButton(text, code)
            btn.setStyleSheet("""
                QPushButton { background-color: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; text-align: left; font-weight: 900; color: #334155; font-size: 12px; }
                QPushButton:hover { background-color: #f1f5f9; border-color: #3b82f6; }
            """)
            btn.clicked.connect(lambda checked, c=code: self.set_main_layer(c))
            layer_vbox.addWidget(btn)
            self.layer_btns.append(btn)
            
        map_controls_layout.addWidget(layer_card)
        
        legend_card = QFrame()
        legend_card.setStyleSheet("background: #f8fafc; border-radius: 12px; border: 1px solid #eef2f7;")
        legend_vbox = QVBoxLayout(legend_card)
        legend_vbox.setContentsMargins(15, 15, 15, 15)
        lbl_l = QLabel("LEJANT")
        lbl_l.setStyleSheet("color: #64748b; font-weight: bold; font-size: 11px; margin-bottom: 5px;")
        legend_vbox.addWidget(lbl_l)
        
        for name, color in self.classes:
            row = QHBoxLayout()
            color_box = QLabel()
            color_box.setFixedSize(12, 12)
            color_box.setStyleSheet(f"background:{color}; border-radius:6px;")
            txt = QLabel(name)
            txt.setStyleSheet("font-size: 12px; font-weight:bold; color: #333;")
            row.addWidget(color_box)
            row.addWidget(txt)
            row.addStretch()
            legend_vbox.addLayout(row)
            
        map_controls_layout.addWidget(legend_card)
        map_controls_layout.addStretch()
        
        map_internal_layout.addWidget(map_controls_widget)
        map_layout.addLayout(map_internal_layout)
        right_layout.addWidget(map_card, stretch=1) 

        # --- SINIF BAZLI DOĞRULUK KARTLARI ---
        metrics_card, metrics_internal_layout = self.create_card_container("", "Sınıf Bazlı Doğruluk Analizi (F1-Skoru)")
        metrics_h_layout = QHBoxLayout()
        metrics_h_layout.setSpacing(15)
        self.metric_cards = {}
        
        self.class_metric_names = [
            ("Mısır Doğruluğu", "🌽", 1), 
            ("Fındık Doğruluğu", "🌰", 2),
            ("Pirinç Doğruluğu", "🌾", 3), 
            ("Kavak Doğruluğu", "🌳", 4),
            ("Bina & Yol Doğr.", "🏙️", 5), 
            ("Su Doğruluğu", "💧", 7)
        ]
        
        for metric, icon, c_idx in self.class_metric_names:
            card = QFrame()
            card.setStyleSheet("background: #f8fafc; border-radius: 14px; border: 1px solid #eef2f7;")
            self.add_shadow(card, radius=10, offset=2, opacity=5)
            
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 15, 10, 15)
            card_layout.setAlignment(Qt.AlignCenter)
            card_layout.setSpacing(8)
            
            title = QLabel(f"{icon}  {metric}")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color:#64748b; font-weight:800; font-size:12px; background: transparent; border:none;")
            
            value = QLabel("-")
            value.setAlignment(Qt.AlignCenter)
            value.setStyleSheet("color:#94a3b8; font-size: 22px; font-weight: 900; background: transparent; border:none;") 
            
            card_layout.addWidget(title)
            card_layout.addWidget(value)
            
            self.metric_cards[metric] = {
                "title_label": title, "value_label": value, "c_idx": c_idx
            }
            metrics_h_layout.addWidget(card)
            
        metrics_internal_layout.addLayout(metrics_h_layout)
        right_layout.addWidget(metrics_card)

        # --- PIE CHART ---
        pie_card, pie_layout = self.create_card_container("", "Tahmin Edilen Alan Dağılımı")
        
        self.pie_series = QPieSeries()
        self.pie_series.setHoleSize(0.45) 
        
        placeholder = self.pie_series.append("Veri Bekleniyor...", 100)
        placeholder.setBrush(QColor("#f1f5f9"))
        placeholder.setBorderColor(QColor("white"))
        
        self.pie_chart = QChart()
        self.pie_chart.addSeries(self.pie_series)
        self.pie_chart.setAnimationOptions(QChart.SeriesAnimations)
        self.pie_chart.legend().setAlignment(Qt.AlignRight)
        self.pie_chart.legend().setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.pie_chart.setBackgroundVisible(False) 
        self.pie_chart.layout().setContentsMargins(0, 0, 0, 0)
        
        self.chart_view = QChartView(self.pie_chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setStyleSheet("background: transparent; border: none;")
        self.chart_view.setMinimumHeight(300) 
        
        pie_layout.addWidget(self.chart_view)
        right_layout.addWidget(pie_card)

        body_layout.addWidget(right_container, stretch=1)
        main_layout.addLayout(body_layout)

    def apply_style(self):
        QApplication.setStyle("Fusion")
        self.setStyleSheet("""
            QMainWindow { background: #f6f8fb; }
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; }
            QFrame#toastFrame { background-color: rgba(30, 41, 59, 0.95); border-radius: 16px; border: 1px solid #0f172a; }
            QLabel#toastLabel { color: white; font-weight: 900; font-size: 15px; background: transparent; }
            QPushButton#toastCloseBtn { background: rgba(255, 255, 255, 0.15); color: white; font-weight: 900; font-size: 20px; border-radius: 18px; padding: 0px; }
            QPushButton#toastCloseBtn:hover { background-color: rgba(255, 255, 255, 0.3); }
        """)

    # ================== YENİ HARİTA FONKSİYONLARI ==================
    def toggle_split_screen(self):
        if self.btn_split.isChecked():
            self.split_image_view.show()
            self.update_image_view(reset_zoom=False)
            
            # EKRANI EŞİT BÖL (%50 - %50)
            total_width = self.map_splitter.width()
            self.map_splitter.setSizes([total_width // 2, total_width // 2])
            
            def match_views():
                self._syncing = True
                self.split_image_view.setTransform(self.image_view.transform())
                self.split_image_view.centerOn(self.image_view.mapToScene(self.image_view.viewport().rect().center()))
                self._syncing = False
            
            QTimer.singleShot(50, match_views)
        else:
            self.split_image_view.hide()
            self.update_image_view(reset_zoom=False)

    def zoom_in_maps(self):
        self.image_view.zoom_in()

    def zoom_out_maps(self):
        self.image_view.zoom_out()

    def set_main_layer(self, code):
        self.left_layer_code = code
        self.update_image_view(reset_zoom=False) # ZOOM SIFIRLANMAZ
        
    def on_layer_dropped(self, code, viewer):
        if viewer == self.image_view:
            self.left_layer_code = code
        elif viewer == self.split_image_view:
            self.right_layer_code = code
        self.update_image_view(reset_zoom=False) # ZOOM SIFIRLANMAZ

    def get_pixmap_for_layer(self, code):
        if code == "rgb": return getattr(self, 'pixmap_rgb', QPixmap())
        elif code == "fc": return getattr(self, 'pixmap_fc', QPixmap())
        elif code == "ndvi": return getattr(self, 'pixmap_ndvi', QPixmap())
        elif code == "ai": return getattr(self, 'pixmap_seg', QPixmap())
        return QPixmap()

    # ZOOM SIFIRLAMA (RESET) KONTROLÜ EKLENDİ
    def update_image_view(self, reset_zoom=False):
        if self.raster_data is None: return
        h, w = self.raster_data.shape[0], self.raster_data.shape[1]
        
        if self.rgb_base is not None:
            rgb_bytes = self.rgb_base.tobytes()
            qimg_rgb = QImage(rgb_bytes, w, h, 3*w, QImage.Format_RGB888).copy()
            self.pixmap_rgb = QPixmap.fromImage(qimg_rgb)
            
        if self.fc_base is not None:
            fc_bytes = self.fc_base.tobytes()
            qimg_fc = QImage(fc_bytes, w, h, 3*w, QImage.Format_RGB888).copy()
            self.pixmap_fc = QPixmap.fromImage(qimg_fc)
            
        if hasattr(self, 'ndvi_rgb') and self.ndvi_rgb is not None:
            ndvi_bytes = self.ndvi_rgb.tobytes()
            qimg_ndvi = QImage(ndvi_bytes, w, h, 3*w, QImage.Format_RGB888).copy()
            self.pixmap_ndvi = QPixmap.fromImage(qimg_ndvi)
            
        self.pixmap_seg = QPixmap(self.pixmap_rgb) if hasattr(self, 'pixmap_rgb') else QPixmap(w, h)
        if hasattr(self, 'mask_overlay') and self.mask_overlay is not None:
            mh, mw = self.mask_overlay.shape[:2]
            mask_bytes = self.mask_overlay.tobytes()
            mask_qimg = QImage(mask_bytes, mw, mh, mw*4, QImage.Format_RGBA8888).copy()
            mask_pixmap = QPixmap.fromImage(mask_qimg)
            painter = QPainter(self.pixmap_seg)
            painter.drawPixmap(0, 0, mask_pixmap)
            painter.end()

        self.image_view.set_image(self.get_pixmap_for_layer(self.left_layer_code))
        if hasattr(self, 'btn_split') and self.btn_split.isChecked():
            self.split_image_view.set_image(self.get_pixmap_for_layer(self.right_layer_code))
        
        # SADECE YENİ HARİTA YÜKLENİRSE ZOOM SIFIRLANIR (EKRANA SIĞDIRILIR)
        if reset_zoom:
            QTimer.singleShot(50, lambda: self.image_view.fitInView(self.image_view.scene.sceneRect(), Qt.KeepAspectRatio))
            if hasattr(self, 'btn_split') and self.btn_split.isChecked():
                QTimer.singleShot(50, lambda: self.split_image_view.fitInView(self.split_image_view.scene.sceneRect(), Qt.KeepAspectRatio))

    def prompt_export_layer(self):
        if self.raster_data is None:
            self.show_toast("İndirilecek harita yok!")
            return
            
        dialog = DownloadDialog(self)
        if dialog.exec_():
            selected_code = dialog.get_selected_layer()
            self.export_layer_to_tif(selected_code)

    def export_layer_to_tif(self, layer_code):
        names = {"ai": "Segmentasyon_Maskesi.tif", "rgb": "Orijinal_RGB.tif", "fc": "Kizilotesi.tif", "ndvi": "NDVI_Bitki.tif"}
        default_name = names.get(layer_code, "Harita.tif")
        
        file_path, _ = QFileDialog.getSaveFileName(self, "TIF Kaydet", default_name, "TIFF Files (*.tif *.tiff)")
        if not file_path: return
            
        self.status_label.setText("💾 Kaydediliyor...")
        QApplication.processEvents()
        
        try:
            h, w = self.raster_data.shape[:2]
            if layer_code == "ai": 
                if not hasattr(self, 'pred_map') or self.pred_map is None:
                    self.show_toast("Önce modeli çalıştırın!")
                    return
                data_to_save = self.pred_map.astype(np.uint8)
                count = 1 
            elif layer_code == "rgb": 
                data_to_save = self.rgb_base; count = 3
            elif layer_code == "fc": 
                data_to_save = self.fc_base; count = 3
            elif layer_code == "ndvi": 
                if not hasattr(self, 'ndvi_rgb'): return
                data_to_save = self.ndvi_rgb; count = 3

            transform = getattr(self, 'geo_transform', None)
            crs = getattr(self, 'geo_crs', None)
            
            with rasterio.open(file_path, 'w', driver='GTiff', height=h, width=w, count=count, dtype=data_to_save.dtype, crs=crs, transform=transform) as dst:
                if count == 1: dst.write(data_to_save, 1)
                else:
                    for i in range(count): dst.write(data_to_save[:, :, i], i + 1)
                        
            self.status_label.setText("✅ TIF Başarıyla Kaydedildi!")
            self.show_toast("Harita başarıyla bilgisayarınıza kaydedildi!")
            
        except Exception as e:
            self.status_label.setText("❌ Kaydetme hatası!")
            self.show_toast(f"HATA: {str(e)}")

    def load_tif(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "TIF Seç", "", "TIFF Files (*.tif *.tiff)")
        if not file_path: return
        try:
            self.status_label.setText("⚙️ Görüntü yükleniyor ve katmanlar hesaplanıyor...")
            QApplication.processEvents()
            
            with rasterio.open(file_path) as src:
                data = src.read()
                self.geo_transform = src.transform 
                self.geo_crs = src.crs 
                self.raster_data = np.transpose(data, (1, 2, 0))
                h, w, c = self.raster_data.shape
                
                rgb = self.raster_data[:, :, [2,1,0]]
                rgb = np.clip(rgb, 0, 2500)
                self.rgb_base = np.ascontiguousarray(((rgb / 2500.0) * 255).astype(np.uint8))
                
                if c >= 7: 
                    fc = self.raster_data[:, :, [6, 2, 1]]
                    fc = np.clip(fc, 0, 3500)
                    self.fc_base = np.ascontiguousarray(((fc / 3500.0) * 255).astype(np.uint8))
                    
                    nir = self.raster_data[:, :, 6].astype(np.float32)
                    red = self.raster_data[:, :, 2].astype(np.float32)
                    denom = (nir + red)
                    denom[denom == 0] = 1 
                    ndvi = (nir - red) / denom
                    ndvi_norm = np.clip((ndvi + 1) / 2.0, 0, 1)
                    
                    r = (255 * (1 - ndvi_norm) * 0.7).astype(np.uint8) 
                    g = (255 * ndvi_norm).astype(np.uint8) 
                    b = np.zeros_like(r)
                    self.ndvi_rgb = np.ascontiguousarray(np.dstack([r, g, b]))
                else:
                    self.fc_base = self.rgb_base
                    self.ndvi_rgb = self.rgb_base

            self.mask_overlay = None
            self.left_layer_code = "rgb"
            
            # SADECE YENİ HARİTA YÜKLENDİĞİNDE ZOOM SIFIRLANIR!
            self.update_image_view(reset_zoom=True) 
            
            base_name = os.path.basename(file_path).split('.')[0]
            gt_dir = os.path.dirname(file_path)
            
            potential_masks = [
                f"{base_name}_tarim_referans.tif", f"{base_name}_tarim_referans.TIF",
                f"{base_name}_referans.tif", f"{base_name}_referans.TIF",
                f"RF_Harita_{base_name}.tif", f"RF_Harita_{base_name}.TIF"
            ]
            
            self.gt_mask = None
            for mask_name in potential_masks:
                gt_path = os.path.join(gt_dir, mask_name)
                if os.path.exists(gt_path):
                    try:
                        with rasterio.open(gt_path) as src_gt:
                            if src_gt.width == w and src_gt.height == h:
                                self.gt_mask = src_gt.read(1).astype(np.uint8)
                                self.show_toast(f"✅ Referans ({mask_name}) Otomatik Bulundu!")
                                break
                    except Exception: pass

            self.status_label.setText("✅ Görüntü hazır. İşleme başlayabilirsiniz.")
            
        except Exception as e:
            self.status_label.setText("❌ Yükleme sırasında hata!")
            self.show_toast(f"HATA: {str(e)}")

    def load_gt(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Manuel Referans Mask", "", "TIFF Files (*.tif *.tiff)")
        if not file_path: return
        try:
            with rasterio.open(file_path) as src:
                self.gt_mask = src.read(1).astype(np.uint8)
            self.show_toast("✅ Referans mask manuel yüklendi.")
            if self.pred_map is not None:
                self.status_label.setText("📊 Metrikler hesaplanıyor...")
                self.evaluate_model()
        except Exception as e:
            self.show_toast(f"Maske Hatası: {str(e)}")

    def run_model(self):
        if self.raster_data is None:
            self.status_label.setText("⚠️ Önce uydu görüntüsü yükleyin!")
            return
        try:
            start_time = time.time()
            display_name = self.model_combo.currentText()
            model_path = f"modeller/{self.models_dict.get(display_name, display_name)}"
            
            self.status_label.setText(f"🚀 Model başlatılıyor...")
            QApplication.processEvents()
            
            fix_keras3_h5(model_path)
            model = tf.keras.models.load_model(model_path, compile=False)
            
            h, w, c = self.raster_data.shape
            PATCH_SIZE = 128
            img_normalized = np.clip(self.raster_data / 10000.0, 0, 1).astype(np.float32)
            pred_map = np.zeros((h, w), dtype=np.uint8)
            conf_map = np.zeros((h, w), dtype=np.float32) 
            
            total_patches = len(range(0, h, PATCH_SIZE)) * len(range(0, w, PATCH_SIZE))
            patch_count = 0
            
            for y in range(0, h, PATCH_SIZE):
                for x in range(0, w, PATCH_SIZE):
                    y1, y2 = y, min(y + PATCH_SIZE, h)
                    x1, x2 = x, min(x + PATCH_SIZE, w)
                    patch = np.zeros((PATCH_SIZE, PATCH_SIZE, c), dtype=np.float32)
                    patch[0:(y2-y1), 0:(x2-x1), :] = img_normalized[y1:y2, x1:x2, :]
                    patch_input = np.expand_dims(patch, axis=0)
                    
                    pred = model.predict(patch_input, verbose=0)
                    pred_probs = pred[0]
                    
                    pred_class = np.argmax(pred_probs, axis=-1)
                    pred_conf = np.max(pred_probs, axis=-1) 
                    
                    pred_map[y1:y2, x1:x2] = pred_class[0:(y2-y1), 0:(x2-x1)]
                    conf_map[y1:y2, x1:x2] = pred_conf[0:(y2-y1), 0:(x2-x1)]
                    
                    patch_count += 1
                    percent = int((patch_count / total_patches) * 100)
                    self.progress.setValue(percent)
                    self.status_label.setText(f"⚡ AI Analizi: %{percent}")
                    QApplication.processEvents()
                    
            self.pred_map = pred_map
            self.conf_map = conf_map
            
            elapsed = round(time.time() - start_time, 2)
            total_pixels = np.sum(pred_map > 0)
            total_ha = total_pixels * 0.01
            
            self.pie_series.clear()
            found_classes = 0
            for idx in range(1, 8):
                class_pixels = np.sum(pred_map == idx)
                if class_pixels > 0:
                    found_classes += 1
                    percentage = (class_pixels / max(1, total_pixels)) * 100
                    name = [n for j, (n, c) in enumerate(self.classes) if (j+1)==idx][0]
                    color = [c for j, (n, c) in enumerate(self.classes) if (j+1)==idx][0]
                    
                    slice_item = self.pie_series.append(f"{name} (%{percentage:.1f} - {class_pixels*0.01:.1f} Ha)", percentage)
                    slice_item.setBrush(QColor(color))
                    slice_item.setBorderColor(QColor("white"))
                    slice_item.setBorderWidth(2)
            
            self.live_stats["Toplam Alan"].setText(f"{total_ha:.1f} Ha")
            self.live_stats["Bulunan Sınıf"].setText(f"{found_classes} Sınıf")
            self.live_stats["Model"].setText(f"{display_name.split(' ')[0]}")
            self.live_stats["İşlem Süresi"].setText(f"{elapsed} sn")
            
            colors = {
                0: [0,0,0,0], 1: [240,196,25,180], 2: [197,123,57,180],
                3: [132,204,22,180], 4: [93,187,99,180], 5: [181,181,181,180], # <-- 3 NUMARAYI GÜNCELLEDİK
                6: [157,204,76,180], 7: [75,134,240,180]
            }
            rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)
            for class_idx, color in colors.items():
                rgba_mask[pred_map == class_idx] = color
                
            self.mask_overlay = np.ascontiguousarray(rgba_mask)
            self.left_layer_code = "ai"
            
            # MODEL BİTTİĞİNDE ZOOMU BOZMA!
            self.update_image_view(reset_zoom=False)
            
            self.status_label.setText(f"✅ Analiz Tamamlandı! ({elapsed} sn)")
            self.evaluate_model()
            
        except Exception as e:
            self.status_label.setText("❌ Hata oluştu.")
            self.show_toast(f"HATA: {str(e)}")

    def evaluate_model(self):
        if self.pred_map is None: return
        
        if self.gt_mask is None:
            self.show_toast("Referans Bulunamadı!\nModelin kendi tespit eminlik (güven) oranları hesaplandı.")
            
            for metric, icon, c_idx in self.class_metric_names:
                card = self.metric_cards[metric]
                class_mask = (self.pred_map == c_idx)
                
                if np.sum(class_mask) > 0:
                    conf = np.mean(self.conf_map[class_mask]) * 100
                    card["value_label"].setText(f"%{conf:.1f}")
                    card["value_label"].setStyleSheet("color:#10b981; font-weight: 900; font-size: 26px; background: transparent;")
                else:
                    card["value_label"].setText("Bölgede Yok")
                    card["value_label"].setStyleSheet("color:#94a3b8; font-weight: 900; font-size: 18px; background: transparent;")
            return
            
        try:
            self.status_label.setText("📊 Sınıf bazlı doğruluklar hesaplanıyor...")
            QApplication.processEvents()
            
            h_pred, w_pred = self.pred_map.shape
            if self.gt_mask.shape != (h_pred, w_pred):
                self.gt_mask = cv2.resize(self.gt_mask, (w_pred, h_pred), interpolation=cv2.INTER_NEAREST)
            y_true = self.gt_mask.flatten()
            y_pred = self.pred_map.flatten()
            
            unique_true_classes = np.unique(y_true)
            class_f1_scores = f1_score(y_true, y_pred, labels=[1, 2, 3, 4, 5, 7], average=None, zero_division=0)
            
            for i, (metric, icon, c_idx) in enumerate(self.class_metric_names):
                card = self.metric_cards[metric]
                
                if c_idx not in unique_true_classes:
                    card["value_label"].setText("Bölgede Yok")
                    card["value_label"].setStyleSheet("color:#94a3b8; font-weight: 900; font-size: 18px; background: transparent;")
                else:
                    val = class_f1_scores[i]
                    if val >= 0.90: val_color = "#10b981" 
                    elif val >= 0.80: val_color = "#f59e0b" 
                    else: val_color = "#ef4444" 

                    card["value_label"].setText(f"%{val*100:.1f}")
                    card["value_label"].setStyleSheet(f"color:{val_color}; font-weight: 900; font-size: 26px; background: transparent;")

            self.show_toast("✅ Sınıf Bazlı Doğruluk Analizi Tamamlandı!\nAna ürünlerin başarı oranları ekrana yansıtıldı.")
            self.status_label.setText("✅ İşlem Tamamlandı.")
            
        except Exception as e:
            self.status_label.setText(f"❌ Test Hatası: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AgriVisionAI()
    window.show()
    sys.exit(app.exec_())
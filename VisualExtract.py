# upload_widget.py
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class UploadWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.init_ui()

    def init_ui(self):
        self.label = QLabel("No image uploaded")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedSize(300, 300)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_button)
        self.setLayout(layout)

    def upload_image(self):
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            pixmap = QPixmap(path)
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def get_image_path(self):
        return self.image_path

from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtCore import Qt

class AnalyzeWidget(QWidget):
    def __init__(self, ai_model, upload_widget):
        super().__init__()
        self.ai_model = ai_model
        self.upload_widget = upload_widget
        self.init_ui()

    def init_ui(self):
        self.result_label = QLabel("Result will appear here")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)

        self.analyze_button = QPushButton("Generate Caption")
        self.analyze_button.clicked.connect(self.analyze_image)

        layout = QVBoxLayout()
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def analyze_image(self):
        image_path = self.upload_widget.get_image_path()
        if image_path:
            caption = self.ai_model.generate_caption(image_path)
            self.result_label.setText(f"🧠 Caption: {caption}")
        else:
            self.result_label.setText("⚠️ Please upload an image first.")

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageCaptionAI:
    def __init__(self):
        # Load pre-trained model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def generate_caption(self, image_path):
        # Open and process image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Generate caption
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QWidget, QFileDialog, QTextEdit
)
from PyQt6.QtGui import QPixmap
from upload_manager import UploadManager
from image_caption_ai import ImageCaptionAI

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Captioning App")
        self.setGeometry(100, 100, 800, 600)

        # Initialize managers
        self.uploader = UploadManager()
        self.captioner = ImageCaptionAI()

        # GUI components
        self.label = QLabel("No Image Uploaded")
        self.label.setScaledContents(True)
        self.label.setFixedSize(400, 400)

        self.caption_box = QTextEdit()
        self.caption_box.setReadOnly(True)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.caption_box)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_image(self):
        image_path = self.uploader.upload_image()
        if image_path:
            pixmap = QPixmap(image_path)
            self.label.setPixmap(pixmap)
            caption = self.captioner.generate_caption(image_path)
            self.caption_box.setPlainText(caption)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

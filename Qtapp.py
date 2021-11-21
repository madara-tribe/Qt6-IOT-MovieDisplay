import sys, os
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from qtWidgets.RightCamWidget import RightCamWidget
from qtWidgets.LeftWidget import LeftWidget

from cfg import Cfg
from solver import load_model
import onnxruntime

class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
 
        cfg = Cfg
        ort_session = onnxruntime.InferenceSession('onnx/cycleGAN_AB.onnx')
        G_AB, G_BA, _, _ = load_model(cfg, device="cpu", model_path='tb')
        
        # RIGHT Side camera widget
        here_path = os.path.dirname(os.path.abspath(__file__))
        self.plot_layout = QVBoxLayout()
        self.video_path = os.path.join(here_path, 'qtWidgets/src/video.mp4')
        self.right_widget = RightCamWidget(self, self.video_path, G_BA, G_AB, ort_session)
        self.plot_layout.addWidget(self.right_widget)
        self.setCentralWidget(self.right_widget)
        
        
        # Left side widget
        cap = self.right_widget.return_cap()
        self.leftDock = QDockWidget("Left Widget", self)
        self.leftside = LeftWidget(self, cap)
        self.leftDock.setWidget(self.leftside)
 
        self.leftDock.setAllowedAreas(Qt.LeftDockWidgetArea
                                   | Qt.RightDockWidgetArea)
        self.leftDock.setFeatures(QDockWidget.DockWidgetMovable
                                  | QDockWidget.DockWidgetFloatable \
                                  #|QDockWidget.DockWidgetVerticalTitleBar)
                                  )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.leftDock)
        
        
        
 
def main():
    app = QApplication(sys.argv)
    # app.setStyle(QStyleFactory.create('Cleanlooks'))
    w = MyMainWindow()
    w.setWindowTitle("PySide Layout on QMainWindow")
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()

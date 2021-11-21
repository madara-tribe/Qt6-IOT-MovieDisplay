from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import sys
import time

class ProgrssBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.botan = QPushButton('Start',self)
        self.botan.setGeometry(10,10,150,50)
        self.botan.pressed.connect(self.osaretara)

    def osaretara(self):
        max = 100
        mess = "Loading now..."
        progressDialog = QProgressDialog(mess, "Cancel", 0, max, self)
        progressDialog.setWindowTitle("Progress Dialog")
         
        for count in range(max+1):
            qApp.processEvents()
            if progressDialog.wasCanceled():
                break
            progressDialog.setValue(count)
            progressDialog.setLabelText(mess + "%d %%" % count)
            time.sleep(0.1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = ProgrssBar()
    main.show()
    app.exec()

        

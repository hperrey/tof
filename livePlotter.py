from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import time
import matplotlib.pyplot as plt


def RTreader(file):
    file.seek(0,2)
    while True:
        line = file.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line


        


app = QtGui.QApplication([])
mw = QtGui.QMainWindow()#main window
mw.resize(600,400)#resize main window
view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
mw.setCentralWidget(view)
mw.show()
mw.setWindowTitle('Signal viewer: ScatterPlot and histogram')
## create area1 to add plot to
w1 = view.addPlot()
#w1.setRange(xRange=[-0, 1029], yRange=[-6, 6])
s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))


def update():
    wave = open("sometext","r")
    wavelines = RTreader(wave)
    row=0
    index=[i for i in range(0,1029)]
    for line in wavelines:
        row+=1
        if row%8==0:
            data=line.split()
            data=[int(i) for i in data]
            print((data))
            #print(type(data[0]))
            s1.addPoints(index,data)
            w1.addItem(s1)
            timer.start(5)#pause for how many miliseconds  
    plt.plot(index,data)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)#pause for how many miliseconds  
## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
















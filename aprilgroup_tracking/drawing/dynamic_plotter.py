from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import numpy as np


class DynamicPlotter():

    def __init__(self, size=(1000,720)):
        # PyQtGraph stuff
        self.app = QtGui.QApplication([])
        self.window = QtGui.QMainWindow()
        self.window.setWindowTitle("DodecaPen Drawing")
        self.window.resize(*size)
        self.start = False
        self.init_points()

        self._generate_ui()
        self.pen_style = QtCore.Qt.SolidLine
        self.pen_colour = (0, 255, 255)

        self.window.show()

    def init_points(self):

        self.points = []

        x1 = np.empty((0,))
        y1 = np.empty((0,))
        z1 = np.empty((0,))

        self.points.extend([x1, y1, z1])

    def setData(self, x, y):
        self.curve.setData(x, y)

    def onNewData(self, x, y):
        pen = pg.mkPen(width=2, color=self.pen_colour, style=self.pen_style)
        self.curve = self.plt.plot([], pen=pen, connect='finite')
        self.setData(x, y)
    
    def start_graph(self):
        self.start = True

    def stop_graph(self):
        self.start = False
        
    def clear_graph(self):
        self.plt.clear()
        self.init_points()

    def save_graph(self):
        # create an exporter instance, as an argument give it
        # the item you wish to export
        exporter = pg.exporters.ImageExporter(self.plt.plotItem)

        # set export parameters if needed
        exporter.parameters()['width'] = 1000   # (note this also affects height parameter)

        # save to file
        exporter.export('aprilgroup_tracking/drawing/dodeca_drawings/dodecaPen_drawing.png')
        
    def _generate_ui(self):
        
        self._create_toolbars()
        self._create_actions()

        # pg.setConfigOption('background', 'm')
        # pg.setConfigOption('foreground', 'k')

        self.plt = pg.plot(title='DodecaPen Drawing Graph')
        # self.plt.resize(*size)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'y image values', 'pixels')
        self.plt.setLabel('bottom', 'x image values', 'pixels')

        self.window.setCentralWidget(self.plt)
  
    def create_spacers(self):
        # spacer widget for left
        left_spacer = QtGui.QWidget()
        left_spacer.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # spacer widget for right
        # you can't add the same widget to both left and right. you need two different widgets.
        right_spacer = QtGui.QWidget()
        right_spacer.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        return left_spacer, right_spacer

    def _create_toolbars(self):
        
        left_spacer, right_spacer = self.create_spacers()
        left_spacer2, right_spacer2 = self.create_spacers()

        # Bottom Toolbar
        menu_toolbar = QToolBar("Bottom Menu")

        self.start_btn = QtGui.QPushButton("Start")
        self.stop_btn = QtGui.QPushButton("Stop")
        self.reset_btn = QtGui.QPushButton("Reset")
        self.save_btn = QtGui.QPushButton("Save")
        self.style_btns(self.start_btn)
        self.style_btns(self.stop_btn)
        self.style_btns(self.reset_btn)
        self.style_btns(self.save_btn)

        menu_toolbar.addWidget(left_spacer)
        menu_toolbar.addWidget(self.start_btn)
        menu_toolbar.addWidget(self.stop_btn)
        menu_toolbar.addWidget(self.reset_btn)
        menu_toolbar.addWidget(self.save_btn)
        menu_toolbar.addWidget(right_spacer)
        self.bottom_tools = self.window.addToolBar(QtCore.Qt.BottomToolBarArea, menu_toolbar)

        # Top Toolbar
        top_toolbar = QToolBar("Top Menu")

        self.select_pencolour = QtGui.QPushButton("Pen Colour")
        self.style_btns(self.select_pencolour)

        # Creating a Drop Down Box
        self.combo_box = pg.ComboBox(self.window)
        # self.style_btns(self.combo_box)
        self.combo_box.setStyleSheet("""
                    text-align: center;
                    font: bold 12px;
                    font-family: Lato;
                    padding: 7px 0px 7px 2px;
                    """)

        # Pen Type List
        pen_type_list = ["Pen Type", "Solid Line", "Dash Line", "Dot Line", "Dash Dot Line", "Dash Dot Dot Line", "Custom Dash Line"]
        # adding list of items to combo box
        self.combo_box.addItems(pen_type_list)
  
        # item
        item ="Pen Type"
        # setting current item
        self.combo_box.setCurrentText(item)

        self.pen_colour_combo = pg.ComboBox(self.window)
        # self.style_btns(self.combo_box)
        self.pen_colour_combo.setStyleSheet("""
                    text-align: center;
                    font: bold 12px;
                    font-family: Lato;
                    padding: 7px 0px 7px 2px;
                    """)

        # Pen Type List
        pen_colour_list = ['Pen Colour', 'Blue', 'Green', 'Red', 'Cyan', 'Magenta', 'Yellow', 'White']
        # adding list of items to combo box
        self.pen_colour_combo.addItems(pen_colour_list)
  
        # item
        pen_colour_item = 'Pen Colour'
        # setting current item
        self.pen_colour_combo.setCurrentText(pen_colour_item)

        top_toolbar.addWidget(left_spacer2)
        top_toolbar.addWidget(self.select_pencolour)
        top_toolbar.addWidget(self.combo_box)
        top_toolbar.addWidget(self.pen_colour_combo)
        top_toolbar.addWidget(right_spacer2)
        self.top_tools = self.window.addToolBar(QtCore.Qt.TopToolBarArea, top_toolbar)

    def _create_actions(self):

        self.start_btn.clicked.connect(self.start_graph)
        self.stop_btn.clicked.connect(self.stop_graph)
        self.reset_btn.clicked.connect(self.clear_graph)
        self.save_btn.clicked.connect(self.save_graph)
        self.select_pencolour.clicked.connect(self.color_picker)
        self.combo_box.activated.connect(self.get_line_style)
        self.pen_colour_combo.activated.connect(self.get_pen_colour)
    
    def color_picker(self):
        color = QtGui.QColorDialog.getColor(parent=self.window)
        print("color", color)

    def get_line_style(self):

        pen_style_mappings = {
            'Solid Line': QtCore.Qt.SolidLine,
            'Dash Line': QtCore.Qt.DashLine,
            'Dot Line': QtCore.Qt.DotLine,
            'Dash Dot Line': QtCore.Qt.DashDotLine,
            'Dash Dot Dot Line': QtCore.Qt.DashDotDotLine,
            'Custom Dash Line': QtCore.Qt.CustomDashLine,
        }
        
        # finding the content of current item in combo box
        self.pen_style = pen_style_mappings[self.combo_box.currentText()]
    
    def get_pen_colour(self):

        pen_colour_mappings = {
            'Blue': 'b',
            'Green': 'g',
            'Red': 'r',
            'Cyan': 'c',
            'Magenta': 'm',
            'Yellow': 'y',
            'White': 'w'
        }

        # finding the content of current item in combo box
        self.pen_colour = pen_colour_mappings[self.pen_colour_combo.currentText()]
  
    def style_btns(self, widget):
        widget.setContentsMargins(2, 2, 2, 2)
        widget.setFixedSize(130, 30)
        widget.setStyleSheet(
        """
        QPushButton {
            background-color: #05445E;
            color: #FFFFFF;
            border-style: outset;
            padding: 2px;
            margin-right:10px;
            text-align: center;
            font: bold 12px;
            font-family: Lato;
            border: none;
        }
        QPushButton:hover {
            background-color: #189AB4;
        }
        """
        )

    def run(self):
        self.app.exec_()

if __name__ == '__main__':

    m = DynamicPlotter()
    m.run()
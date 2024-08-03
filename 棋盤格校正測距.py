import PySimpleGUI as psg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import MouseEvent, MouseButton
import cv2
import numpy as np

class MyMap:
    def __init__(self):
        self.img = None
        self.pixel_resolution = None
        self.chessboard_size = (9, 6)  # 内部角點
        self.widget()
        self.win = psg.Window('demo', layout=self.layout(), location=(10, 10), finalize=True)
        self.tk_fig = FigureCanvasTkAgg(self.fig, self.win['canvas'].TKCanvas)
        self.tool = NavigationToolbar2Tk(self.tk_fig, self.win['canvas'].TKCanvas)
        self.tk_fig.get_tk_widget().pack()
        self.event = None
        self.value = None

    def widget(self):
        self.tt = psg.Text('分辨率: ')
        self.np_ratio = psg.Input(size=(25, 1), default_text='1')
        self.cvs = psg.Canvas(key='canvas')
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.mpl_connect('button_press_event', self.coordinate)
        self.bt = psg.Button(key='draw', button_text='draw')
        self.np_start = psg.Input(readonly=True, size=(25, 1))
        self.np_end = psg.Input(readonly=True, size=(25, 1))
        self.np_result = psg.Input(readonly=True, size=(25, 1))
        self.bt_file = psg.Input(key='file_input', readonly=True, size=(25, 1))
        self.bt_browse = psg.FileBrowse(key='Browse', target='file_input')
        self.bt_calibrate = psg.Button(key='calibrate', button_text='calibrate')
        self.actual_width = psg.Input(key='chessboard_square_size', size=(25, 1), default_text='10.000') 
        self.func = {
            'draw': self.img_show,
            'calibrate': self.calibrate
        }

    def calibrate(self):
        try:
            square_size = float(self.value['chessboard_square_size'])
            
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                #精確率與迭代次數指定
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(self.img, self.chessboard_size, corners, ret)
                
                self.fig.clear()
                fig_img = self.fig.add_subplot(111)
                fig_img.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
                self.tk_fig.draw()

                # 兩角點的像素距離
                pixel_distance = np.linalg.norm(corners[0] - corners[1])
                
                # 每像素實際距離
                self.pixel_resolution = square_size / pixel_distance
                psg.popup(f"每像素代表的實際距離: {self.pixel_resolution:.3f} mm")
            else:
                psg.popup_error("未檢測到棋盤格角點")
        except Exception as e:
            psg.popup_error(f"校準時發生錯誤: {e}")

    def cpu_distance(self, x1, y1, x2, y2):
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance

    def coordinate(self, e: MouseEvent):
        print(e.dblclick, e.button)
        if e.dblclick:
            if e.button == MouseButton.LEFT:
                self.np_start.update(value=f'start-{e.xdata:.2f}-{e.ydata:.2f}')
            elif e.button == MouseButton.RIGHT:
                self.np_end.update(value=f'end-{e.xdata:.2f}-{e.ydata:.2f}')
                x1, y1 = self.np_start.get().split('-')[1:]
                x2, y2 = self.np_end.get().split('-')[1:]
                distance = self.cpu_distance(float(x1), float(y1), float(x2), float(y2))
                if self.pixel_resolution:
                    distance = self.pixel_resolution * distance
                self.np_result.update(value=f'result:{distance:.3f} mm')

    def layout(self):
        layout = [
            [self.bt_file, self.bt_browse, self.bt, self.bt_calibrate, self.tt, self.actual_width, self.np_ratio, self.np_start, self.np_end, self.np_result],
            [self.cvs]
        ]
        return layout

    def img_show(self):
        try:
            file_path = self.value['file_input']
            print(f"Selected file path: {file_path}")
            if not file_path:
                psg.popup_error("請先選擇圖片文件")
                return
            self.img = cv2.imread(file_path)
            if self.img is None:
                print(f"cv2.imread 無法讀取圖像文件: {file_path}")
                psg.popup_error("無法讀取圖片文件")
                return
            print(f"Image shape: {self.img.shape}")
            self.fig.clear()
            fig_img = self.fig.add_subplot(111)
            fig_img.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            self.tk_fig.draw()
        except Exception as e:
            psg.popup_error(f"發生錯誤: {e}")

    def show(self):
        while True:
            try:
                self.event, self.value = self.win.read()
                print(f"Event: {self.event}, Value: {self.value}")
                if self.event in [None, 'Exit']:
                    print("Closing the window")
                    break
                if self.event in self.func:
                    print(f"Calling function for event: {self.event}")
                    self.func[self.event]()
            except Exception as e:
                psg.popup_error(f"發生錯誤: {e}")
        self.win.close()

if __name__ == '__main__':
    app = MyMap()
    app.show()

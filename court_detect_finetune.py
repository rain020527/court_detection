
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


class Params(object):

    class Bar(object):
        def __init__(self, idx, name, vmin, vmax, vinit, vstep, func, facecolor='lightgoldenrodyellow'):
            super(Params.Bar, self).__init__()
            self.idx = idx
            axbar = plt.axes([0.25, 0.2-0.03*self.idx, 0.65,
                              0.01], facecolor=facecolor)
            self.sbar = Slider(axbar, name, vmin, vmax,
                               valinit=vinit, valstep=vstep)
            self.sbar.on_changed(func)
            self.param = vinit

        def get_bar(self):
            return self.sbar

    def __init__(self, imgobj, img):
        super(Params, self).__init__()
        self.imgobj = imgobj
        self.img = img
        self.gray = adjust_gamma(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0.4)
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0).astype(int)
        self.sub = self.gray.astype(int) - blur
        self.bars = {}

    def add_bar(self, name, vmin, vmax, vinit, vstep, func):
        idx = len(self.bars)+1
        self.bars[name] = self.Bar(idx, name, vmin, vmax, vinit, vstep, func)
        return self.bars[name].get_bar()

    def get_param(self, name):
        return self.bars[name].param

    def update_param(self, name, val):
        self.bars[name].param = val

    def add_reset(self, func):
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
        button.on_clicked(func)
        return button


def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()


def add_slider(init, unit, axcolor):
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    global sfreq, samp
    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=init[0], valstep=unit[0])
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=init[1], valstep=unit[1])
    sfreq.on_changed(update)
    samp.on_changed(update)
    return sfreq, samp


def reset(event):
    sfreq.reset()
    samp.reset()


def add_reset(axcolor):
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    button.on_clicked(reset)
    return button


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()


def add_selector(axcolor):
    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
    radio.on_clicked(colorfunc)
    return radio


def save_param(event):
    pass


def add_save(axcolor):
    pass


def main2():
    global fig
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    global t
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    delta_f = 5.0
    s = a0 * np.sin(2 * np.pi * f0 * t)
    global l
    l, = plt.plot(t, s, lw=2)
    ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    sfreq, samp = add_slider(init=[f0, a0], unit=[
                             delta_f, 1.0], axcolor=axcolor)
    button = add_reset(axcolor=axcolor)
    radio = add_selector(axcolor=axcolor)

    plt.show()
    pass


def main():

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img = cv2.cvtColor(cv2.imread('6_avg.png'), cv2.COLOR_BGR2RGB)
    imgobj = plt.imshow(img)

    # global all_params
    all_params = Params(imgobj, img)

    def flip_horizon(val):
        if val % 2 == 0:
            all_params.imgobj.set_data(all_params.img)
        else:
            all_params.imgobj.set_data(np.flip(all_params.img, 1))
        fig.canvas.draw_idle()

    def flip_virtical(val):
        if val % 2 == 0:
            all_params.imgobj.set_data(all_params.img)
        else:
            all_params.imgobj.set_data(np.flip(all_params.img, 0))
        fig.canvas.draw_idle()

    def _lowThreshold(val):
        detected_edges = np.clip(
            all_params.gray.astype(int)+all_params.sub*2, 
            a_min=0, a_max=255).astype('uint8')
        detected_edges = cv2.Canny(
            detected_edges, 
            val, 
            val*3, 
            apertureSize=3)
        dst = cv2.bitwise_and(
            all_params.img, 
            all_params.img, 
            mask=detected_edges)
        lines = cv2.HoughLinesP(
            detected_edges, 
            1, np.pi/180, 
            int(all_params.get_param('threshold')), 
            np.array([]), 
            all_params.get_param('min_line_length'), 
            all_params.get_param('max_line_gap'))
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(dst, (x1, y1), (x2, y2), (254, 0, 0), 1)
        all_params.imgobj.set_data(dst)
        all_params.update_param('lowThreshold', val)
        fig.canvas.draw_idle()

    def _threshold(val):
        detected_edges = np.clip(
            all_params.gray.astype(int)+all_params.sub*2, 
            a_min=0, a_max=255).astype('uint8')
        detected_edges = cv2.Canny(
            detected_edges, 
            all_params.get_param('lowThreshold'), 
            all_params.get_param('lowThreshold')*3, 
            apertureSize=3)
        dst = cv2.bitwise_and(
            all_params.img, 
            all_params.img, 
            mask=detected_edges)
        lines = cv2.HoughLinesP(
            detected_edges, 
            1, np.pi/180, 
            int(val), 
            np.array([]), 
            all_params.get_param('min_line_length'), 
            all_params.get_param('max_line_gap'))
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(dst, (x1, y1), (x2, y2), (254, 0, 0), 1)
        all_params.imgobj.set_data(dst)
        all_params.update_param('threshold', val)
        fig.canvas.draw_idle()

    def _min_line_length(val):
        detected_edges = np.clip(
            all_params.gray.astype(int)+all_params.sub*2, 
            a_min=0, a_max=255).astype('uint8')
        detected_edges = cv2.Canny(
            detected_edges, 
            all_params.get_param('lowThreshold'), 
            all_params.get_param('lowThreshold')*3, 
            apertureSize=3)
        dst = cv2.bitwise_and(
            all_params.img, 
            all_params.img, 
            mask=detected_edges)
        lines = cv2.HoughLinesP(
            detected_edges, 
            1, np.pi/180, 
            int(all_params.get_param('threshold')), 
            np.array([]), 
            val, 
            all_params.get_param('max_line_gap'))
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(dst, (x1, y1), (x2, y2), (254, 0, 0), 1)
        all_params.imgobj.set_data(dst)
        all_params.update_param('min_line_length', val)
        fig.canvas.draw_idle()

    def _max_line_gap(val):
        detected_edges = np.clip(
            all_params.gray.astype(int)+all_params.sub*2, 
            a_min=0, a_max=255).astype('uint8')
        detected_edges = cv2.Canny(
            detected_edges, 
            all_params.get_param('lowThreshold'), 
            all_params.get_param('lowThreshold')*3, 
            apertureSize=3)
        dst = cv2.bitwise_and(
            all_params.img, 
            all_params.img, 
            mask=detected_edges)
        lines = cv2.HoughLinesP(
            detected_edges, 
            1, np.pi/180, 
            int(all_params.get_param('threshold')), 
            np.array([]), 
            all_params.get_param('min_line_length'), 
            val)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(dst, (x1, y1), (x2, y2), (254, 0, 0), 1)
        all_params.imgobj.set_data(dst)
        all_params.update_param('max_line_gap', val)
        fig.canvas.draw_idle()

    # bar1 = all_params.add_bar('Hflip', 0, 10, 0, 1, func=flip_horizon)
    # bar2 = all_params.add_bar('Vflip', 0, 10, 0, 1, func=flip_virtical)
    bar1 = all_params.add_bar('lowThreshold',    0, 100, 85, 1, func=_lowThreshold)
    bar2 = all_params.add_bar('threshold',       0,  50, 30, 1, func=_threshold)
    bar3 = all_params.add_bar('min_line_length', 0, 100, 75, 1, func=_min_line_length)
    bar4 = all_params.add_bar('max_line_gap',    0,  20,  5, 1, func=_max_line_gap)

    def reset(event):
        bar1.reset()
        bar2.reset()
        bar3.reset()
        bar4.reset()
    rbutton = all_params.add_reset(reset)

    plt.show()


if __name__ == '__main__':
    main()

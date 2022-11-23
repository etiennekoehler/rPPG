from scipy import signal as scipy
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from statistics import *
import math as m
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
import sys
from PyQt5.QtWidgets import QApplication
from yarppg.ui import MainWindow
from yarppg.rppg import RPPG
from yarppg.rppg.processors import ColorMeanProcessor, FilteredProcessor, PosProcessor
from yarppg.rppg.hr import HRCalculator
from yarppg.rppg.filters import get_butterworth_filter
from yarppg.ui.cli import (get_detector, get_mainparser, get_processor,
                           parse_frequencies)
from yarppg.rppg.roi.roi_detect import FaceMeshDetector

#participant1_start_times   = ["01:00:30:00", "01:01:54:00", "01:03:24:00", "01:04:54:00", "01:06:36:00", "01:08:12:00", "01:09:50:00", "01:11:28:00", "01:12:54:00", "01:14:20:00"]
#participant1_2_start_times = ["01:16:30:00", "01:20:00:00", "01:21:36:00"] -01:16:08:03
#participant2_start_times   = ["01:00:30:00", "01:01:58:00", "01:03:24:00", "01:04:50:00", 01:06:26:00", "01:07:56:00", "01:09:22:00", "01:10:48:00", "01:12:18:00", "01:13:44:00"]
#participant2_2_start_times = ["01:15:36:00", "01:19:36:00", "01:21:32:00"] -01:15:07:00
#participant3_start_times   = ["00:12:08:00", "00:13:20:00", "00:14:38:00", "00:15:48:00", "00:17:26:00", "00:18:50:00", "00:20:22:00", "00:21:46:00", "00:23:06:00", "00:24:24:00", "00:25:48:00", "00:28:14:00", "00:30:10:00"]
#participant4_start_times = ["01:00:34:00", "01:02:08:00", "01:03:26:00", "01:05:02:00", "01:06:32:00", "01:07:52:00", "01:09:34:00", "01:10:52:00", "01:12:16:00", "01:14:04:00", "01:15:52:00", "01:18:28:00", "01:20:56:00"]
#participant5_start_times = ["01:00:30:00", "01:02:56:00", "01:04:26:00", "01:05:52:00", "01:07:12:00", "01:08:28:00", "01:09:42:00", "01:11:02:00", "01:12:18:00", "01:13:44:00", "01:15:08:00", "01:16:28:00", "01:17:48:00"]



def downsample(signal, time, new_sample_rate):
    wave_duration = time[-1]
    samples = len(signal)
    sample_rate = samples / wave_duration
    q = sample_rate / new_sample_rate
    q_rd = m.ceil(q)
    samples_decimated = int(samples / q)

    print('sample_rate:', sample_rate)
    print('sample_quotient:', q)
    print('sample_decimated:', samples_decimated)

    signal_downsampled = scipy.decimate(signal, q_rd)
    xnew = np.linspace(0, wave_duration, samples_decimated, endpoint=False)

    # shorten xnew to match pulse_downsampled
    xnew = xnew[:len(signal_downsampled)]

    return (signal_downsampled, xnew)

def get_peaks(signal):
    peaks_indices, _ = find_peaks(signal, distance=7)
    peaks = [i in peaks_indices for i in range(len(signal))]
    return peaks, peaks_indices

def calculate_IBIs(peaks, frame_rate, filter_width):
    # count frames between peaks, convert frame-count to milliseconds
    IBIs = []
    counter = 0
    start = False
    for elem in peaks:
        if not elem and not start:
            start = True
        if not elem and start:
            counter += 1
        elif elem and start:
            milliseconds = (counter / frame_rate) * 1000
            IBIs.append(milliseconds)
            counter = 0
    return remove_outliers(IBIs, filter_width)

def remove_outliers(IBIs, filter_width):
    start_length = len(IBIs)

    # remove intervals smaller than 250ms and bigger than 2000ms
    IBIs = [i for i in IBIs if i > 250 and i < 2000]

    if len(IBIs) < 2: return IBIs

    mean_IBIs = mean(IBIs)
    stdev_IBIs = stdev(IBIs)

    # remove intervals further than 'filter_width' standard deviations from the mean
    IBIs  = [i for i in IBIs if i > mean_IBIs - filter_width * stdev_IBIs and i < mean_IBIs + filter_width * stdev_IBIs]

    return IBIs

def remove_outliers_in_interval(IBIs, IBIs_in_interval, filter_width):
    print(len(IBIs_in_interval))
    # remove intervals smaller than 250ms and bigger than 2000ms
    IBIs_in_interval = [i for i in IBIs_in_interval if 250 < i < 2000]

    mean_IBIs = mean(IBIs)
    stdev_IBIs = stdev(IBIs)

    # remove intervals further than 'filter_width' standard deviations from the mean
    IBIs_in_interval = [i for i in IBIs_in_interval if
                        mean_IBIs - filter_width * stdev_IBIs < i < mean_IBIs + filter_width * stdev_IBIs]
    print(len(IBIs_in_interval))
    return IBIs_in_interval

def calculate_HR(peaks, win_size, frame_rate, stride):
    # win_size: number of seconds taken to calculate HR
    HR = []
    current_HR = 0
    stride_counter = 0
    index = 0
    win_size_samples = win_size * frame_rate

    if win_size < 2:
        raise IOError("win_size for HR calculation is too small")

    while index < len(peaks):
        if stride_counter > stride:
            stride_counter = 0
        else:
            index += 1
            stride_counter += 1
            HR.append(current_HR)
            continue

        if index < win_size_samples-1:
            index += 1
            continue

        IBIs = calculate_IBIs(peaks[index - win_size_samples:index], frame_rate, 3)
        if IBIs != []:
            avg_IBI = mean(IBIs)
            current_HR = (60 / avg_IBI) * 1000

        HR.append(current_HR)
        index += 1
        stride_counter += 1

    return HR

def calculate_HR_with_IBIs(IBIs, win_size, frame_rate, stride_ms):
    win_size_ms = win_size * 1000
    length_video = sum(IBIs)
    counter_ms = 0
    timestamps = []
    HR = []

    if win_size < 2:
        raise IOError("win_size for HR calculation is too small")

    while counter_ms < length_video:

        if counter_ms < win_size_ms:
            counter_ms += stride_ms
            continue

        IBIs_in_window = get_IBIs_in_interval(IBIs=IBIs, interval=[counter_ms - win_size_ms, counter_ms])
        IBIs_in_window_filtered = remove_outliers(IBIs_in_window, 2)

        if len(IBIs_in_window_filtered) < 5:
            counter_ms += stride_ms
            continue

        avg_IBI = mean(IBIs_in_window_filtered)

        HR.append((60 / avg_IBI) * 1000)
        timestamps.append(counter_ms)

        counter_ms += stride_ms

    return HR, timestamps


def get_IBIs_in_interval(IBIs, interval):
    total_time = 0
    IBIs_index = 0
    IBIs_in_interval = []

    while (True):
        if (total_time > interval[1]):
            break

        if (total_time > interval[0]):
            IBIs_in_interval.append(IBIs[IBIs_index-1])

        total_time += IBIs[IBIs_index]
        IBIs_index += 1
    return IBIs_in_interval


def calculate_HRV_rmssd(IBIs):
    index = 0
    sum = 0

    if len(IBIs) < 2: return 0

    while index < len(IBIs)-1:
        difference = IBIs[index] - IBIs[index+1]
        squared_difference = (difference) ** 2
        sum += squared_difference
        index += 1
    avg = sum / len(IBIs)
    return m.sqrt(avg)

def calculate_HRV_rmssd_shortterm(peaks, win_size, stride, frame_rate):
    index = 0
    stride_counter = 0
    HRVs = []
    current_HRV = 0

    if win_size < 2:
        raise IOError("win_size for HRV calculation is too small")

    while index < len(peaks) - 1:
        if stride_counter > stride:
            stride_counter = 0
        else:
            index += 1
            stride_counter += 1
            HRVs.append(current_HRV)
            continue

        if index < win_size-1:
            index += 1
            continue

        IBIs = calculate_IBIs(peaks=peaks[index-win_size:index], frame_rate=frame_rate, filter_width=1)
        HRV = calculate_HRV_rmssd(IBIs)
        HRVs.append(HRV)
        current_HRV = HRV
        index += 1
        stride_counter += 1
    return HRVs

def compute_MAE_HR(signal_pred, signal_gt, win_size, frame_rate, stride):
    peaks_gt, _ = get_peaks(signal_gt)
    peaks_pred, _ = get_peaks(signal_pred)

    HR_gt = calculate_HR(peaks=peaks_gt, win_size=win_size, frame_rate=frame_rate, stride=stride)
    HR_pred = calculate_HR(peaks=peaks_pred, win_size=win_size, frame_rate=frame_rate, stride=stride)

    print(len(signal_pred))
    print(len(signal_gt))

    mae = mean_absolute_error(HR_gt, HR_pred)
    return mae, HR_gt, HR_pred

def compute_MAE_HR_2(HR_gt, HR_pred):

    mae = mean_absolute_error(HR_gt, HR_pred)
    return mae

def compute_MAE_HRV(signal_pred, signal_gt, win_size, stride, metric, frame_rate):
    peaks_gt, _ = get_peaks(signal_gt)
    peaks_pred, _ = get_peaks(signal_pred)

    HRV_gt = calculate_HRV_rmssd_shortterm(peaks=peaks_gt, win_size=win_size, stride=stride, frame_rate=frame_rate)
    HRV_pred = calculate_HRV_rmssd_shortterm(peaks=peaks_pred, win_size=win_size, stride=stride, frame_rate=frame_rate)

    #HRV_gt = [x * 1000 for x in HRV_gt]
    #HRV_pred = [x * 1000 for x in HRV_pred]

    mae = mean_absolute_error(HRV_gt, HRV_pred)

    return mae, HRV_gt, HRV_pred

def getDataframePeaks(all_gt_ibis):
    #change all_ms to 30 per sec, then for each peak check which timecode is closest -> asign it
    allDataframes = []
    for gt_ibis in all_gt_ibis:
        index = 0
        start_time_ms = []
        while (index < len(gt_ibis) - 1):
            sum_before = sum(gt_ibis[0:index])
            start_time_ms.append(sum_before)
            index += 1
        start_time_ms = [int(i) for i in start_time_ms]

        all_ms = list(range(0, start_time_ms[-1] + 1))
        is_peak = [i in start_time_ms for i in range(all_ms[-1] + 1)]

        df = {'time_ms': all_ms, 'peak': is_peak}
        peak_df = pd.DataFrame(data=df)
        allDataframes.append(peak_df)
    return allDataframes


def split_df(start_times, df_hr):
    "01:00:30:00"
    one_minute_slices = []
    start_times_ms = [read_timecode(i) for i in start_times]
    for start_time in start_times_ms:
        one_minute_slice = df_hr[start_time:(start_time+60000)]
        one_minute_slices.append(one_minute_slice)
    return one_minute_slices

def create_df(frame_rate: int, length: int):
    # interval: 30
    # dont sownsample,
    df = {'second': all_ms, 'frame': is_peak}

def split_IBIS(IBIs, start_times_ms):
    start_times_index = 0
    IBIs_index = 0
    slices_start_indexes = []
    slices_end_indexes = []
    total_time = 0
    looking_for_start = True
    slice_length = 0


    while(start_times_index < len(start_times_ms) or not looking_for_start):
        if slice_length > 60000 and (not looking_for_start):
            slices_end_indexes.append(IBIs_index-1)
            slice_length = 0
            looking_for_start = True
            IBIs_index += 1
        elif  looking_for_start and total_time > start_times_ms[start_times_index]:
            slices_start_indexes.append(IBIs_index)
            looking_for_start = False
            slice_length = total_time - start_times_ms[start_times_index]
            start_times_index += 1
            IBIs_index += 1
        else:
            IBIs_index += 1
            slice_length += IBIs[IBIs_index]
        total_time += IBIs[IBIs_index]

    if len(slices_end_indexes) != len(slices_start_indexes):
        raise IOError("Error in splitIBIS")

    slices = []
    index = 0
    while(index < len(slices_start_indexes)):
        slices.append(IBIs[slices_start_indexes[index] : slices_end_indexes[index]])
        index+=1

    return slices

def read_timecode(timecode_string):
    # timecode_string: "01:00:30:00"
    cs = int(timecode_string[9:11])
    s = int(timecode_string[6:8])
    m = int(timecode_string[3:5])
    ms = (((m*60)+s)*1000)+cs
    return ms

def strech_HR_pred(time_gt, time_pred):
    end_time = time_gt[-1]
    start_time = time_gt[0]

    step = round((end_time - start_time) / len(time_pred))
    new_time_pred = []
    counter = start_time

    for x in range(0, len(time_pred) - 2):
        counter += step
        new_time_pred.append(counter)

    new_time_pred = [start_time] + new_time_pred + [time_gt[-1]]
    return new_time_pred

def resample_HR_pred(time_gt, time_pred, HR_pred):
    new_time_pred = strech_HR_pred(time_gt, time_pred)
    f = interp1d(new_time_pred, HR_pred)
    new_HR_pred = []
    for time in time_gt:
        HR = f(time)
        HR = HR.tolist()
        new_HR_pred.append(HR)
    return new_HR_pred

def run_rPPG(input_file, output_file):
    parser = get_mainparser()
    app = QApplication(sys.argv)

    roi_detector = FaceMeshDetector(draw_landmarks=False)

    digital_lowpass = get_butterworth_filter(30, 1.5)
    hr_calc = HRCalculator(parent=app, update_interval=30, winsize=300,
                           filt_fun=lambda vs: [digital_lowpass(v) for v in vs])

    winsize = 250
    processor = PosProcessor(winsize)

    cutoff = parse_frequencies('0.5,2')
    if cutoff is not None:
        digital_bandpass = get_butterworth_filter(30, cutoff, "bandpass")
        processor = FilteredProcessor(processor, digital_bandpass)

    # 'cohface/1/3/data.avi'
    rppg = RPPG(roi_detector=roi_detector,
                video=input_file,
                hr_calculator=hr_calc,
                parent=app,
                )
    rppg.add_processor(processor)

    for c in "rgb":
        rppg.add_processor(ColorMeanProcessor(channel=c, winsize=1))

    rppg.output_filename = output_file

    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(1000, 400),
                     legend=True,
                     graphwin=300,
                     blur_roi=-1,
                     )
    for i in range(3):
        win.set_pen(index=i + 1, color="rgb"[i], width=1)

    x = win.execute()
    return

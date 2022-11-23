from pulse_comparison import *
import pandas as pd
from numpy import genfromtxt
import math

def run_post_pipeline(participant, condition, win_size, stride_ms, frame_rate):

    IBIs_pred = get_IBIs_pred(participant, condition)

    IBIs_gt = get_IBIs_gt(participant, condition)

    HR_pred, time_pred = calculate_HR_with_IBIs(IBIs=np.asarray(IBIs_pred), win_size=win_size, frame_rate=frame_rate, stride_ms=stride_ms)
    HR_gt, time_gt = calculate_HR_with_IBIs(IBIs=IBIs_gt, win_size=win_size, frame_rate=frame_rate, stride_ms=stride_ms)

    #resample HR_pred to values of time_gt
    new_HR_pred = resample_HR_pred(time_gt, time_pred, HR_pred)

    mae = compute_MAE_HR_2(HR_gt, new_HR_pred)

    std, distances = compute_std(new_HR_pred, HR_gt, mae)

    return new_HR_pred, HR_gt, time_gt, mae, std

def get_IBIs_pred(participant, condition):
    path = "my_signals/{}/{}/data.csv".format(participant, condition)
    signal_pred_df = pd.read_csv(path)
    signal_pred_df = signal_pred_df.rename(columns={"ts": "time", "p0": "pulse_pred"})

    peaks_pred, peaks_pred_indices = get_peaks(signal_pred_df['pulse_pred'])
    signal_pred_df.insert(2, 'peaks_pred', peaks_pred, False)
    IBIs_pred = calculate_IBIs(peaks=signal_pred_df['peaks_pred'], frame_rate=30, filter_width=1)

    return IBIs_pred

def get_IBIs_gt(participant, condition):
    participant1_start_times = ["01:00:30:00", "01:01:54:00", "01:03:24:00", "01:04:54:00", "01:06:36:00",
                                "01:08:12:00", "01:09:50:00", "01:11:28:00", "01:12:54:00", "01:14:20:00"]
    participant1_2_start_times = ["01:00:21:57", "01:03:52:00", "01:05:27:57"]
    participant2_start_times = ["01:00:30:00", "01:01:58:00", "01:03:24:00", "01:04:50:00", "01:06:26:00",
                                "01:07:56:00", "01:09:22:00", "01:10:48:00", "01:12:18:00", "01:13:44:00"]
    participant2_2_start_times = ["01:00:29:00", "01:04:29:00", "01:06:25:00"]
    participant3_start_times = ["00:12:08:00", "00:13:20:00", "00:14:38:00", "00:15:48:00", "00:17:26:00",
                                "00:18:50:00", "00:20:22:00", "00:21:46:00", "00:23:06:00", "00:24:24:00",
                                "00:25:48:00", "00:28:14:00", "00:30:10:00"]
    participant4_start_times = ["01:00:34:00", "01:02:08:00", "01:03:26:00", "01:05:02:00", "01:06:32:00",
                                "01:07:52:00", "01:09:34:00", "01:10:52:00", "01:12:16:00", "01:14:04:00",
                                "01:15:52:00", "01:18:28:00", "01:20:56:00"]
    participant5_start_times = ["01:00:30:00", "01:02:56:00", "01:04:26:00", "01:05:52:00", "01:07:12:00",
                                "01:08:28:00", "01:09:42:00", "01:11:02:00", "01:12:18:00", "01:13:44:00",
                                "01:15:08:00", "01:16:28:00", "01:17:48:00"]
    participant1and2_start_times = [participant1_start_times, participant1_2_start_times, participant2_start_times, participant2_2_start_times]
    all_participant_start_times = [participant1_start_times, participant2_start_times, participant3_start_times, participant4_start_times, participant5_start_times]
    if participant < 3:
        if condition < 11:
            path = "my_groundtruth/Participant{}_1.csv".format(participant)
            if participant == 1:
                participant_start_times = participant1_start_times
            else:
                participant_start_times = participant2_start_times
            participant_start_times_ms = [read_timecode(i) for i in participant_start_times]
        else:
            path = "my_groundtruth/Participant{}_2.csv".format(participant)
            if participant == 1:
                participant_start_times = participant1_2_start_times
            else:
                participant_start_times = participant2_2_start_times
            participant_start_times_ms = [read_timecode(i) for i in participant_start_times]
    else:
        path = "my_groundtruth/Participant{}.csv".format(participant)
        participant_start_times_ms = [read_timecode(i) for i in all_participant_start_times[participant - 1]]

    IBIs_gt_entire = genfromtxt(path, delimiter=',')
    IBIs_gt_entire = IBIs_gt_entire[1:]
    IBIs_gt_entire_slices = split_IBIS(IBIs_gt_entire, participant_start_times_ms)
    if participant < 3 and condition > 10:
        IBIs_gt = IBIs_gt_entire_slices[condition - 11]
    else:
        IBIs_gt = IBIs_gt_entire_slices[condition - 1]

    return IBIs_gt

def compute_std(HR_pred, HR_gt, mae):
    deviation_acc_sq = 0
    distances = []
    index = 0
    while index < len(HR_gt):
        distance = abs(HR_gt[index] - HR_pred[index])
        deviation_sq = pow(distance - mae, 2)
        deviation_acc_sq += deviation_sq
        distances.append(distance)
        index+=1
    var = deviation_acc_sq / index
    std = math.sqrt(var)
    return std, distances

def get_colors(nums):

    colors = []
    for num in nums:
        if num == 1:
            colors.append("#CC665C")
        elif num == 2:
            colors.append("#C0DA6B")
        elif num == 3:
            colors.append("#7DD89A")
        elif num == 4:
            colors.append("#6182D5")
        else:
            colors.append("#BB5ED4")
    return colors



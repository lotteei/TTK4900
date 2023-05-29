from help_func.h2s_experiment_info import get_h2s_level, get_time_for_bad_behaviour
from help_func.load_pickle import get_pos_data_dicts
from datetime import timedelta as td
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


# File 
save_path = 'data/'
file_name = 'behaviour_data.csv'

# Sliding window parameters
sliding_window = td(minutes=10)
step_size = td(minutes=5)
step = td(seconds=1)
bins_count = 50

bool_save = False


date_list = ['2022_06_27/', '2022_06_28/', '2022_06_29/', '2022_06_30/', '2022_07_01/', '2022_07_02/', '2022_07_03/', '2022_07_04/', '2022_07_05/', '2022_07_06/', '2022_07_07/']
data_path = '/home/lotte/TTK4900/Stereo_RCNN/results/tracker/all_files/' #+ date_list[day] 
start = [td(0, 15, 0, 0, 36, 15),td(0, 0, 0, 0, 18, 7), td(0, 0, 0, 0, 0, 7), td(0, 0, 0, 0, 0, 7), td(0, 0, 0, 0, 45, 6), td(0, 0, 0, 0, 40, 6),td(0, 0, 0, 0, 45, 6), td(0, 0, 0, 0, 45, 6), td(0, 0, 0, 0, 45, 6), td(0, 0, 0, 0, 48, 6), td(0, 0, 0, 0, 42, 6)]



def add_data_to_set(dict_tot_vel,dict_tot_decomp_vel, dict_tot_sr, date):
    bad_b = get_time_for_bad_behaviour(date)
    start_time = bad_b[0]
    end_time = bad_b[1] 

    len_min = 999999
    len_max = 0
    len_avg = 0
    number = 0

    final_list = []
    whiskers = []
    time_list = []

    while (start_time <= end_time):

        temp_v_list = []
        temp_xv_list = []
        temp_yv_list = []
        temp_zv_list = []
        temp_sr_list = []
  
        iterate_date = start_time
        while start_time < iterate_date + sliding_window:
           
            try:
            
                temp_v_list.extend(dict_tot_vel[start_time])
                temp_xv_list.extend(dict_tot_decomp_vel[start_time]['x'])
                temp_yv_list.extend(dict_tot_decomp_vel[start_time]['y'])
                temp_zv_list.extend(dict_tot_decomp_vel[start_time]['z'])
                temp_sr_list.extend(dict_tot_sr[start_time])

            except:
                i=0

            start_time += step   
        

        all_lists= [temp_v_list, temp_xv_list, temp_yv_list, temp_zv_list, temp_sr_list ]
        n_list = []
        bins_list = []
        
        for l in all_lists:
            n, bins = np.histogram(l, bins=bins_count, density=True)
            n_list.append(n)
            bins_list.append(bins)
            len_avg += len(l)
            number += 1
            if len(l) < len_min:
                len_min = len(l)

            if len(l) > len_max:
                len_max = len(l)

       
        data = {'Date': iterate_date,
                'n_v': [n_list[0]],
                'bins_v': [bins_list[0][1:]],
                'n_xv': [n_list[1]],
                'bins_xv': [bins_list[1][1:]],
                'n_yv': [n_list[2]],
                'bins_yv': [bins_list[2][1:]],
                'n_zv': [n_list[3]],
                'bins_zv': [bins_list[3][1:]],
                'n_a_v': [n_list[4]],
                'bins_a_v': [bins_list[4][1:]],
                'h2s': get_h2s_level(date)}

        
        if bool_save:
            save_data_to_csv(data)
        else:
            plot_histograms(all_lists)

        try:
            final_list.append(sum(temp_v_list)/len(temp_v_list))
            time_list.append(start_time)
        except:
            i=0

        

        start_time -= step_size
    
   
    return final_list, whiskers, time_list


def plot_histograms(lists):

    y_labels = ['$|v|$', '$v_{x}$', '$v_{y}$', '$v_{z}$', '$r_s$']
    colors = ['green', 'steelblue', 'steelblue', 'steelblue', 'orange']

    # Velocity plot
    fig_vel, axes_vel = plt.subplots(4, sharex=True, sharey=True, figsize=(18,15))
    fig_vel.suptitle('Velocity Distribution', fontweight='bold', fontsize=22)

    sns.histplot(lists[1], label=y_labels[1], kde=False, color=colors[1], stat='density', bins=bins_count, ax=axes_vel[0])
    sns.histplot(lists[2], label=y_labels[2], kde=False, color=colors[1],stat='density', bins=bins_count, ax=axes_vel[1])
    sns.histplot(lists[3], label=y_labels[3], kde=False, color=colors[1],stat='density', bins=bins_count, ax=axes_vel[2])
    sns.histplot(lists[0], label=y_labels[0], kde=False, color=colors[0],stat='density', bins=bins_count, ax=axes_vel[3])
    
    axes_vel[3].set(xlabel='(mm/s)')
    
    for i in range(len(axes_vel)):
        axes_vel[i].yaxis.label.set_size(18)
        axes_vel[i].xaxis.label.set_size(18)
        axes_vel[i].legend(fontsize=14)
        
    plt.show()



    # Acceleration plot
    fig_sr, axes_sr = plt.subplots(figsize=(18,5))
    fig_sr.suptitle('Rate of Change of Speed Distribution', fontweight='bold', fontsize=22)

    sns.histplot(lists[4], label=y_labels[4], kde=False, color=colors[4], stat='density', bins=bins_count)
    
    axes_sr.set(xlabel='(mm/s²)')
    axes_sr.yaxis.label.set_size(18)
    axes_sr.xaxis.label.set_size(18)
    axes_sr.legend(fontsize=14)
        
    plt.show()


def save_data_to_csv(data):
    df = pd.DataFrame(data)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        df.to_csv(save_path + file_name, mode='a', index=False, header=True)
    else:
        df.to_csv(save_path + file_name, mode='a', index=False, header=False)


for i in range(0, len(date_list)):
    dict_tot_3Dpos, dict_tot_vel, dict_tot_decomp_vel, dict_tot_sr = get_pos_data_dicts(data_path+date_list[i],  start[i])
    date = date_list[i].replace('_', '/')[:-1]
    print('DATE: ', date)
    add_data_to_set(dict_tot_vel,dict_tot_decomp_vel, dict_tot_sr, date)
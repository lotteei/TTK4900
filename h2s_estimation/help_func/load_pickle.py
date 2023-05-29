import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import timedelta as td
from datetime import timezone
import numpy as np
import pickle
import math




def create_dicts():
        
    global dict_tot_3Dpos
    global dict_tot_vel
    global dict_tot_decomp_vel
    global dict_tot_sr


    dict_tot_3Dpos = {}
    dict_tot_vel = {}
    dict_tot_decomp_vel = {}
    dict_tot_sr = {}


def loadall(path, file):
    with open(path + file + '.pickle', 'rb') as handle:
        while True:
            try:
                yield pickle.load(handle)
            except:
                break



def add_pos_dict(time, value):
    
    if time not in dict_tot_3Dpos.keys():
        dict_tot_3Dpos[time] = {}
        dict_tot_3Dpos[time]['x'] = [value[0]]
        dict_tot_3Dpos[time]['y'] = [value[1]]
        dict_tot_3Dpos[time]['z'] = [value[2]]

    else:
        dict_tot_3Dpos[time]['x'].append(value[0])
        dict_tot_3Dpos[time]['y'].append(value[1])
        dict_tot_3Dpos[time]['z'].append(value[2])
    
def frame_to_dt(date, start, frame):
    
    sec = int(frame[5:])/15
    response = dt.datetime.fromtimestamp(sec, timezone(td(hours=0))) + start
    datetime = dt.datetime(int(date[:4]), int(date[5:-4]), int(date[8:-1]), response.hour,response.minute,response.second )
    

    return datetime


def add_comp_vel_dict(time, value):
    if time not in dict_tot_decomp_vel.keys():
        dict_tot_decomp_vel[time] = {}
        dict_tot_decomp_vel[time]['x'] = [value[0]]
        dict_tot_decomp_vel[time]['y'] = [value[1]]
        dict_tot_decomp_vel[time]['z'] = [value[2]]

    else:
        
        dict_tot_decomp_vel[time]['x'].append(value[0])
        dict_tot_decomp_vel[time]['y'].append(value[1])
        dict_tot_decomp_vel[time]['z'].append(value[2])


def add_vel_dict(time, value):
    if time not in dict_tot_vel.keys():
        dict_tot_vel[time] = {}
        dict_tot_vel[time] = [value]

    else:
        dict_tot_vel[time].append(value)


def add_sr_dict(time, value):
    if time not in dict_tot_sr.keys():
        dict_tot_sr[time] = {}
        dict_tot_sr[time] = [value]

    else:
        dict_tot_sr[time].append(value)


   


def get_pos_data_dicts(path, start):
    date = path[66:-1] + '_'
    print(date)
    create_dicts()
    

    pickle_file_list = ['3Dpos', 'decomp_vel', 'vel', 'acc']
    for pickle_file in pickle_file_list:
        for value in loadall(path + date, pickle_file):
            for frame, values in value.items():
               
                
                datetime = frame_to_dt(date, start, frame)
                # print(datetime)
                
                for key in values:
                    if pickle_file == '3Dpos':
                        add_pos_dict(datetime, values[key])
                    
                    elif pickle_file == 'decomp_vel':
                        add_comp_vel_dict(datetime, values[key]) 

                    elif pickle_file == 'vel':
                        add_vel_dict(datetime, values[key])

                    elif pickle_file == 'acc':
                        add_sr_dict(datetime, values[key])

                    else:
                        print('Something went wrong?')
        



    return dict_tot_3Dpos, dict_tot_vel, dict_tot_decomp_vel, dict_tot_sr  


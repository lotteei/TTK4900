from datetime import datetime, timedelta
import csv

data_path = "/home/lotte/TTK4900/h2s_experiment/aqs-05-045-water_calc_2022-06-28_2022-07-07.csv"

def removeNull(row):
    return ['0' if i=='' else i for i in row]

# Measurements from H2S experiment
time = list()
alkalinity_mgl = list()
co2_mgl = list() 
h2s_ugl = list()
o2_mgl = list() 
o2_sat = list()

s = "2022-06-28T01:02:52.251631Z"[:-8]
s_new =s.replace("T", " ")
datetime_object = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')



with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for i, row in enumerate(csv_reader):
                if (row[3]==''):
                    continue

                row = removeNull(row)
                if i > 1:
                    time.append(datetime.strptime(row[0][:-8], '%Y-%m-%dT%H:%M:%S') + timedelta(hours=2))
                    alkalinity_mgl.append(float(row[1])) 
                    co2_mgl.append(float(row[2])) 
                    h2s_ugl.append(float(row[3])) 
                    o2_mgl.append(float(row[4])) 
                    o2_sat.append(float(row[5])) 


time_for_dates = {'2022/06/27': [datetime(2022, 6, 27, 16, 0), datetime(2022, 6, 27, 20, 51)],
                '2022/06/28': [datetime(2022, 6, 28, 4, 7, 35), datetime(2022, 6, 28, 14, 15, 50)],
                '2022/06/29': [datetime(2022, 6, 29, 4, 31, 38), datetime(2022, 6, 29, 13, 46, 40)],
                '2022/06/30': [datetime(2022, 6, 30, 5, 19, 12), datetime(2022, 6, 30, 15, 11, 19)],
                '2022/07/01': [datetime(2022, 7, 1, 4, 23, 48), datetime(2022, 7, 1, 12, 47, 30)],
                '2022/07/02': [datetime(2022, 7, 2, 4, 35, 48), datetime(2022, 7, 2, 12, 52, 41)],
                '2022/07/03': [datetime(2022, 7, 3, 5, 4, 17), datetime(2022, 7, 3, 15, 3, 43)],
                '2022/07/04': [datetime(2022, 7, 4, 4, 43, 34), datetime(2022, 7, 4, 11, 48, 47)],
                '2022/07/05': [datetime(2022, 7, 5, 3, 47, 6), datetime(2022, 7, 5, 17, 44, 38)],
                '2022/07/06': [datetime(2022, 7, 6, 5, 30, 27), datetime(2022, 7, 6, 11, 37, 29)],
                '2022/07/07': [datetime(2022, 7, 7, 6, 43, 3), datetime(2022, 7, 7, 20, 20, 40)]}

time_for_bad_behaviour = {'2022/06/27': [datetime(2022, 6, 27, 16, 0), datetime(2022, 6, 27, 19, 51)],
                '2022/06/28': [datetime(2022, 6, 28, 8, 33, 0), datetime(2022, 6, 28, 8,55, 0)],
                '2022/06/29': [datetime(2022, 6, 29, 8, 9, 0), datetime(2022, 6, 29, 8, 30, 0)],
                '2022/06/30': [datetime(2022, 6, 30, 8, 15, 0), datetime(2022, 6, 30, 8, 35, 0)],
                '2022/07/01': [datetime(2022, 7, 1, 8, 9, 0), datetime(2022, 7, 1, 9, 0, 0)],
                '2022/07/02': [datetime(2022, 7, 2, 8, 14, 0), datetime(2022, 7, 2, 9, 0, 1)],
                '2022/07/03': [datetime(2022, 7, 3, 8, 0, 0), datetime(2022, 7, 3, 9, 0, 0)],
                '2022/07/04': [datetime(2022, 7, 4, 8, 10, 34), datetime(2022, 7, 4, 9, 00, 47)],
                '2022/07/05': [datetime(2022, 7, 5, 8, 3, 6), datetime(2022, 7, 5, 8, 57, 38)],
                '2022/07/06': [datetime(2022, 7, 6, 8, 6, 0), datetime(2022, 7, 6, 8, 53, 00)],
                '2022/07/07': [datetime(2022, 7, 7, 8, 3, 0), datetime(2022, 7, 7, 9, 35, 0)]}

h2s_level = {'2022/06/27': 0,
                '2022/06/28': 3.5,
                '2022/06/29': 8,
                '2022/06/30': 9.5,
                '2022/07/01': 5.8,
                '2022/07/02': 3.5,
                '2022/07/03': 7.2,
                '2022/07/04': 14.4,
                '2022/07/05': 32,
                '2022/07/06': 67.7,
                '2022/07/07': 66.4}

time_h2s_added = {'2022/06/28': datetime(2022, 6, 28, 8, 25, 0),
                '2022/06/29': datetime(2022, 6, 29, 8, 3, 0),
                '2022/06/30': datetime(2022, 6, 30, 8, 3, 0),
                '2022/07/01': datetime(2022, 7, 1, 8, 00, 0),
                '2022/07/02': datetime(2022, 7, 2, 8, 5, 0),
                '2022/07/03': datetime(2022, 7, 3, 7, 52, 0),
                '2022/07/04': datetime(2022, 7, 4, 8, 00, 0),
                '2022/07/05': datetime(2022, 7, 5, 7, 50, 0),
                '2022/07/06': datetime(2022, 7, 6, 7, 50, 0),
                '2022/07/07': datetime(2022, 7, 7, 7, 50, 0)}


def get_h2s_level(date):
     return h2s_level[date]

def get_time_for_bad_behaviour(date):
    return time_for_bad_behaviour[date]

def get_h2s_meas(date):
    start = time.index(time_for_dates[date][0])
    end = time.index(time_for_dates[date][1])
    return time[start:end], h2s_ugl[start:end]

def get_h2s_add_time(date):
     return time_h2s_added[date]
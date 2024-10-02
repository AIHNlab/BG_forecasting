import os
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as etree
import joblib
from dataloaders.dataloader import Dataloader
from utils import calculate_total_cob, calculate_total_iob

class DataloaderOhio(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)


    def load_data(self):
        train_path = os.path.join(self.directory_path, "OhioT1DM-training")
        train_files = glob(train_path + os.sep + "*.xml")
        for file in train_files:
            df, patient_id = self._get_dataframe(file)
            self.train_dataframes[patient_id] = df 
            self.all_dataframes[patient_id] = df
        
        test_path = os.path.join(self.directory_path, "OhioT1DM-testing")
        test_files = glob(test_path + os.sep + "*.xml")
        for file in test_files:
            df, patient_id = self._get_dataframe(file)
            self.test_dataframes[patient_id] = df 
            self.all_dataframes[patient_id] = df

        #return self.all_dataframes, self.train_dataframes, self.test_dataframes
    
    def _get_dataset_specific_metadata(self):

        if "Ohio2020" in self.directory_path:
            metadata = self._get_2020_dict()
        else:
            metadata = self._get_2018_dict()
        
        for patient_id in metadata.keys():
            self.train_metadata[patient_id + "-ws-training"] = metadata[patient_id]
        for patient_id in metadata.keys():
            self.test_metadata[patient_id + "-ws-testing"] = metadata[patient_id]
        #self._dataframe_metadata()
        #Make the ohio dataloader a subclass of the dataloader class
        #Call a method in the Dataloader that just calculates high level statistics from that person (sampling_rate, number_of_samples, missing_values, features)
        #Could also include some high level relevant statistics like time in range hypoglycemia, hyperglycemia, etc.

        #Also add metadata that is specific to the Ohio dataset (age, sex, diabetes type, etc.)
        
        #This should be a dict which is then saved as a json file per participant
        #return self.train_metadata, self.test_metadata

    def _get_dataframe(self, file, calculate_iob = True):
        columns = ['cbg', 'finger', 'basal', 'hr', 'gsr', 'carbInput', 'temp_basal', 'bolus']
        xmlkeys = ["glucose_level", "finger_stick", "basal", "basis_heart_rate", "basis_gsr", "meal", "temp_basal", "bolus"]

        data_dict = {}
        patient_id = file.split(os.sep)[-1].split('.')[0]
        tree = etree.parse(file)

        #patient_id = fff.split('.')[0]
        #tree = etree.parse(os.path.join(root_dir, fff))
        finaltime = []
        # loop through outpus
        num = 0
        for x in xmlkeys:
            time = []
            val = []
            val2 = []
            time2 = []
            rawtime = []
            # loop through instancies
            for f in tree.iter(x):
                # actual instances loop

                for g in f:
                    # divide time by 300 to get 5 minute intervals
                    if num < 5:
                        val.append(float(g.items()[1][1]))
                        time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)  # 5min a 60 sek = 300
                        rawtime.append(g.items()[0][1])
                    if num == 5:
                        val.append(float(g.items()[2][1]))
                        time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                    if num == 6:  # temp_basal
                        val.append(float(g.items()[2][1]))
                        time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                        time2.append(pd.to_datetime(g.items()[1][1], dayfirst=True).timestamp() / 300)
                    if num == 7:
                        val.append(float(g.items()[3][1]))
                        time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                        time2.append(pd.to_datetime(g.items()[1][1], dayfirst=True).timestamp() / 300)
            if len(time) == 0:
                if num == 6:  # temp_basal
                    num = num + 1
                    continue;
            time = np.array(time)
            val = np.array(val)
            sorter = np.argsort(time)
            time = time[sorter]
            val = val[sorter]

            if num > 5:
                time2 = np.array(time2)
                time2 = time2[sorter]

            # get basetime according to glucose-values
            if num == 0:
                # if 'test' in fff:
                #     joblib.dump(rawtime, '../' + outdir + '/' + fff[:3] + '.timestamps.pkl')
                basetime = np.linspace(time.copy()[0], time.copy()[-1] + 1,
                                    int(time.copy()[-1] + 1 - time.copy()[0]))  # -time.copy()[0]
                data_dict['5minute_intervals_timestamp'] = basetime
                zerotime = time.copy()[0]
                out = np.array(val)
            # do interpolation
            time = np.array(time) - zerotime
            val = np.array(val)
            out = np.full(len(basetime), np.nan)
            # for basal and basal 0s, use carry forward imputation
            if num == 2:
                for i in range(len(time)):
                    if int(time[i]) < len(basetime):
                        out[int(time[i]):] = val[i]
            # basal 0s just shows when the pump is off so update basal array
            elif num == 6:  # temp_basal indicates it the pump was shut down; correct the forward imputed basal-array
                out = data_dict['basal']
                time2 = np.array(time2) - zerotime
                for i in range(len(time)):
                    if int(time[i]) < len(basetime):
                        out[int(time[i]):int(time2[i])] = val[i]
            # For other variables, just put each value at the closest 5 minute time point.
            else:
                for i in range(len(time)):
                    if int(time[i]) < len(basetime) and int(time[i]) >= 0:  # check for "int(time[i]) >= 0" since timestamps from other parameters prior to the first timestamp from glucose_values are igrnored
                        out[int(time[i])] = val[i]
                        if num == 0:

                            # mask for missing cbg data
                            missing_cbg = (np.isnan(out)).astype(float)
                            data_dict['missing_cbg'] = missing_cbg

            # add to dictionary

            if num == 6:
                data_dict['basal'] = out
            else:
                data_dict[columns[num]] = out
            # move onto next.
            num = num + 1
        time = []
        for t in basetime:
            time.append(pd.to_datetime(int(t) * 300, unit='s'))
        data_dict['time'] = time
        df = pd.DataFrame(data_dict)
        df.set_index('5minute_intervals_timestamp')
        print(df.head())
        if calculate_iob == True:
            df['iob'] = calculate_total_iob(df['bolus'].values, ts_min=5, t_action_max_min=240)
            df['iob'] = pd.Series(df['iob']).rolling(window=12, min_periods=1).mean().to_numpy()
            df['cob'] = calculate_total_cob(df['carbInput'].values, carb_absorption=0.8, ts_min=5, t_action_max_min=240)
            df['cob'] = pd.Series(df['cob']).rolling(window=12, min_periods=1).mean().to_numpy()
        return df, patient_id

    def _get_2020_dict(self):
        metadata2020 = {
            '540': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 20,
                'age_range_high': 40,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_630g',
                'sensor_band': 'empatica_embrace',
                'sampling_rate': 300
            },
            '544': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'empatica_embrace',
                'sampling_rate': 300
            },
            '552': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 20,
                'age_range_high': 40,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_630g',
                'sensor_band': 'empatica_embrace',
                'sampling_rate': 300
            },
            '567': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 20,
                'age_range_high': 40,
                'biological_sex': 'female',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_630g',
                'sensor_band': 'empatica_embrace',
                'sampling_rate': 300
            },
            '584': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'empatica_embrace',
                'sampling_rate': 300
            },
            '596': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 60,
                'age_range_high': 80,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'empatica_embrace',
                'sampling_rate': 300
            }
        }
        return metadata2020

    def _get_2018_dict(self):
        metadata2018 = {
            '559': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'female',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'basis_peak',
                'sampling_rate': 300
            },
            '563': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'basis_peak',
                'sampling_rate': 300
            },
            '570': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'male',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'basis_peak',
                'sampling_rate': 300
            },
            '575': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'female',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'basis_peak',
                'sampling_rate': 300
            },
            '588': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'female',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'basis_peak',
                'sampling_rate': 300
            },
            '591': {
                'number_of_samples': 0,
                'number_of_cgm_samples': 0,
                'age_range_low': 40,
                'age_range_high': 60,
                'biological_sex': 'female',
                'diagnosis_type': 'type1',
                'cgm_type': 'medtronic_530g',
                'sensor_band': 'basis_peak',
                'sampling_rate': 300
            }
        }
        return metadata2018
    
if False:
    import xml.etree.ElementTree as etree
    import pandas as pd
    import joblib
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # call this code from the directory that contains the data

    # outdir = 'data2020'
    # os.makedirs('../' + outdir)

    columns = ['cbg', 'finger', 'basal', 'hr', 'gsr', 'carbInput', 'temp_basal',
            'bolus']


    xmlkeys = ["glucose_level", "finger_stick", "basal", "basis_heart_rate", "basis_gsr",
            "meal", "temp_basal", "bolus", ]

    dict = {}

    partition = 'test'
    if partition == 'train':
        root_dir = '../Ohio2020_XML/OhioT1DM-training'  # '../Ohio_XML/OhioT1DM-training'
    elif partition == 'test':
        root_dir = '../Ohio2020_XML/OhioT1DM-testing'  # '../Ohio_XML/OhioT1DM-testing'

    for fff in os.listdir(root_dir):
        if not fff.endswith('.xml'):
            continue
        patient_id = fff.split('.')[0]
        tree = etree.parse(os.path.join(root_dir, fff))
        finaltime = []
        # loop through outpus
        num = 0
        for x in xmlkeys:
            time = []
            val = []
            val2 = []
            time2 = []
            rawtime = []
            # loop through instancies
            for f in tree.iter(x):
                # actual instances loop

                for g in f:
                    # divide time by 300 to get 5 minute intervals
                    if num < 5:
                        val.append(float(g.items()[1][1]))
                        time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)  # 5min a 60 sek = 300
                        rawtime.append(g.items()[0][1])
                    if num == 5:
                        val.append(float(g.items()[2][1]))
                        time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                    if num == 6:  # temp_basal
                        val.append(float(g.items()[2][1]))
                        time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                        time2.append(pd.to_datetime(g.items()[1][1], dayfirst=True).timestamp() / 300)
                    if num == 7:
                        val.append(float(g.items()[3][1]))
                        time.append(pd.to_datetime(g.items()[0][1], dayfirst=True).timestamp() / 300)
                        time2.append(pd.to_datetime(g.items()[1][1], dayfirst=True).timestamp() / 300)
            if len(time) == 0:
                if num == 6:  # temp_basal
                    num = num + 1
                    continue;
            time = np.array(time)
            val = np.array(val)
            sorter = np.argsort(time)
            time = time[sorter]
            val = val[sorter]

            if num > 5:
                time2 = np.array(time2)
                time2 = time2[sorter]

            # get basetime according to glucose-values
            if num == 0:
                # if 'test' in fff:
                #     joblib.dump(rawtime, '../' + outdir + '/' + fff[:3] + '.timestamps.pkl')
                basetime = np.linspace(time.copy()[0], time.copy()[-1] + 1,
                                    time.copy()[-1] + 1 - time.copy()[0])  # -time.copy()[0]
                dict['5minute_intervals_timestamp'] = basetime
                zerotime = time.copy()[0]
                out = np.array(val)
            # do interpolation
            time = np.array(time) - zerotime
            val = np.array(val)
            out = np.full(len(basetime), np.nan)
            # for basal and basal 0s, use carry forward imputation
            if num == 2:
                for i in range(len(time)):
                    if int(time[i]) < len(basetime):
                        out[int(time[i]):] = val[i]
            # basal 0s just shows when the pump is off so update basal array
            elif num == 6:  # temp_basal indicates it the pump was shut down; correct the forward imputed basal-array
                out = dict['basal']
                time2 = np.array(time2) - zerotime
                for i in range(len(time)):
                    if int(time[i]) < len(basetime):
                        out[int(time[i]):int(time2[i])] = val[i]
            # For other variables, just put each value at the closest 5 minute time point.
            else:
                for i in range(len(time)):
                    if int(time[i]) < len(basetime) and int(time[i]) >= 0:  # check for "int(time[i]) >= 0" since timestamps from other parameters prior to the first timestamp from glucose_values are igrnored
                        out[int(time[i])] = val[i]
                        if num == 0:
                            # mask for missing cbg data
                            missing_cbg = (np.isnan(out)).astype(float)
                            dict['missing_cbg'] = missing_cbg

            # add to dictionary
            if num == 6:
                dict['basal'] = out
            else:
                dict[columns[num]] = out
            # move onto next.
            num = num + 1

        # save data frame
        df = pd.DataFrame(dict)
        df.set_index('5minute_intervals_timestamp')
        # save data frame
        df.to_csv(path_or_buf='../Ohio2020_processed/{}/{}_processed.csv'.format(partition, patient_id), index=False)
        # df.to_csv(path_or_buf='../Ohio_processed/{}/{}_processed.csv'.format(partition, patient_id), index=False)
        # if 'test' in fff:
        #     joblib.dump(df, '../' + outdir + '/' + fff[:3] + '.test.pkl')
        # if 'train' in fff:
        #     joblib.dump(df, '../' + outdir + '/' + fff[:3] + '.train.pkl')

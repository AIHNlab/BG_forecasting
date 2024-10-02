import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dataloaders.dataloader import Dataloader
from utils import calculate_total_cob, calculate_total_iob


class DataloaderTidepoolSAP100(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)

    def load_data(self):
        train_path = os.path.join(self.directory_path, "Tidepool-JDRF-SAP100-train", "train-data")
        train_files = glob(train_path + os.sep + "*.csv")
        for file in tqdm(train_files):
            df, patient_id = self._get_dataframe(file)
            self.train_dataframes[patient_id] = df 
            self.all_dataframes[patient_id] = df
        
        test_path = os.path.join(self.directory_path, "Tidepool-JDRF-SAP100-test", "Tidepool-JDRF-SAP100-test","test-data")
        test_files = glob(test_path + os.sep + "*.csv")

        for file in tqdm(test_files):
            df, patient_id = self._get_dataframe(file)
            self.test_dataframes[patient_id] = df 
            self.all_dataframes[patient_id] = df

        return self.all_dataframes, self.train_dataframes, self.test_dataframes

    def _get_dataset_specific_metadata(self):
        train_metadata_path = os.path.join(self.directory_path, "Tidepool-JDRF-SAP100-train", "SAP100-train-metadata-summary.csv")
        test_metadata_path = os.path.join(self.directory_path, "Tidepool-JDRF-SAP100-test", "Tidepool-JDRF-SAP100-test","SAP100-test-metadata-summary.csv")
        train_metadata_from_file = self._get_metadata_from_file(train_metadata_path)
        test_metadata_from_file = self._get_metadata_from_file(test_metadata_path)

        self.train_metadata = self._standardize_metadata(train_metadata_from_file)
        self.test_metadata = self._standardize_metadata(test_metadata_from_file)
        #file_name,firstDate,lastDate,cgmPumpDayStart,cgmPumpDayEnd,cgmPumpDaySpan,cgmPumpDaysWithData,cgmPumpPctSpanWithData,ageStart,ageEnd,yearsLivingWithDiabetesStart,yearsLivingWithDiabetesEnd,diagnosisType,biologicalSex
        #test_c3e1f727f9f9b1e47bd3ee85f78321ddd8a508d6e0765edba1f689023fe6f605.csv,5/10/18,11/28/19,8/5/19,11/2/19,90,90,1.0,4,4,3.0,3.0,type1,
        #test_e9f834f4c3701dc1364a7692f1ead4babaf990211742009c28fa72cd7c74b087.csv,5/11/18,11/15/19,8/18/19,11/15/19,90,90,1.0,5,5,4.0,4.0,,
        #test_65d013a5477dada4f1c10b7ea92befaeefd04c1f8faf7ee104adfe51c48a33a8.csv,11/2/17,11/27/19,4/1/19,6/29/19,90,90,1.0,35,35,17.0,18.0,,male
        #test_b8f98c7fb599335bb374b0b036724bfc714404581d5fa2b75e714b5f642010e6.csv,1/16/18,11/27/19,8/30/19,11/27/19,90,90,1.0,54,54,52.0,52.0,type1,male
        #test_baf43b5cdb97c109223bbde6063c6632a4781a27d01f3c40024b2948bc1dc729.csv,10/6/17,11/5/19,8/17/19,11/4/19,80,80,1.0,39,39,12.0,12.0,,male
        #test_ced97802f9e98a099af9e17747741a14df923483f595959a0b419e802f42643a.csv,10/1/15,11/29/19,8/4/18,11/1/18,90,90,1.0,49,50,18.0,19.0,,male
        #test_9036ac5808b094bc027a31a39e9e8514dad2ad43074fd528283ba3f8ca586fee.csv,10/1/15,9/17/19,8/29/18,11/26/18,90,90,1.0,13,13,3.0,3.0,,female
        #test_ed0d568a49546d64a23613a87ba5bc766477fe187781ef90b779fd5695c85c52.csv,5/28/17,10/30/19,4/2/19,6/30/19,90,90,1.0,5,6,3.0,3.0,,female
        #test_83ce9ebcbeb53a3dfc0899d3bad758902b7f29453a6baf7ebdd46bed2442a8cc.csv,3/24/16,10/11/19,11/3/18,4/4/19,153,73,0.477124183,40,40,30.0,30.0,,male
        #test_e0e0c1b4e8c754ee772f5226162800723cfedd98eff97ae50c8e58b037071610.csv,3/13/16,8/14/19,3/7/19,6/4/19,90,90,1.0,5,5,3.0,3.0,type1,male
        #test_55c5eac723fb563a3f67906b98fe8a90485afd889e9946241a633f51f35e875b.csv,3/6/17,11/30/19,6/6/18,9/3/18,90,90,1.0,43,44,29.0,29.0,,male
        #test_fc2b12608d03f91a22838fa8f7b7a03314c3071d8b373dd94c214e9f3c012992.csv,5/31/17,11/3/19,8/4/19,11/1/19,90,90,1.0,55,55,18.0,18.0,,
        #test_f8ac6861f2848add8df7b32788c05b54b063dec9d24d5726fd1a22b59a6fb90d.csv,1/16/17,12/3/19,6/16/18,9/13/18,90,90,1.0,5,5,1.0,1.0,,female
        #test_777eb0f5a6877c8058352063e1e22400f10bed58e05a610c1be6d32f6f947c6b.csv,9/20/15,2/22/19,10/19/18,1/16/19,90,88,0.977777778,32,32,3.0,3.0,,
        #test_a71abd4e08ab19da1e090bc24f209bc92ec2b55fb028c1b5d100759b26ce34c1.csv,8/19/15,11/25/19,7/28/18,10/25/18,90,90,1.0,58,58,22.0,22.0,,
        #test_dbcb6083fae7a69dd4475687e85061031aaa1d1d45826676cf2ee504648eef49.csv,7/14/18,11/20/19,8/22/19,11/19/19,90,88,0.977777778,36,36,24.0,24.0,type1,female
    
    def _standardize_metadata(self,metadata_from_file):
        standardized_metadata = {}
        for patient_id in metadata_from_file.keys():
            standardized_metadata[patient_id] = {
                'age_range_low': metadata_from_file[patient_id]['ageStart'],
                'age_range_high': metadata_from_file[patient_id]['ageEnd'],
                'biological_sex': metadata_from_file[patient_id]['biologicalSex'],
                'diagnosis_type': metadata_from_file[patient_id]['diagnosisType'],
                #'cgm_type': 'medtronic_630g',
                #'sensor_band': 'empatica_embrace',
                'years_living_with_diagnosis': metadata_from_file[patient_id]['yearsLivingWithDiabetesStart'],
                'sampling_rate': 300
            }
        return standardized_metadata
            
        

    def _get_metadata_from_file(self,metadata_path):
        # Define the column names
        column_names = ['file_name', 'firstDate', 'lastDate', 'cgmPumpDayStart', 'cgmPumpDayEnd', 'cgmPumpDaySpan', 
                        'cgmPumpDaysWithData', 'cgmPumpPctSpanWithData', 'ageStart', 'ageEnd', 'yearsLivingWithDiabetesStart', 
                        'yearsLivingWithDiabetesEnd', 'diagnosisType', 'biologicalSex']

        # Read the file into a DataFrame
        df = pd.read_csv(metadata_path, names=column_names)

        # Initialize an empty dictionary to store the metadata
        metadata = {}

        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            # Get the file name without the extension
            file_name = row['file_name'].split('.')[0]

            # Convert the row into a dictionary and store it in the metadata dictionary
            metadata[file_name] = row.to_dict()

        # Return the metadata dictionary
        return metadata
    
    def _get_dataframe(self, file, calculate_iob = True):
        dict={}
        patient_id = file.split(os.sep)[-1].split('.')[0]
        print(patient_id)
        read_data_raw = pd.read_csv(file,
                                    usecols=['carbInput',  # carbohydrate input at pump calculation
                                            'deliveryType',  # if scheduled basal rate is active or suspended
                                            'insulinCarbRatio',  # insulinCarbRatio at pump calculation
                                            'insulinOnBoard',  # IOB at pump calculation
                                            'normal',  # amount of bolus insulin
                                            'rate',  # active basal rate (check "deliveryType" for timesteps where basal rate is suspended
                                            'time',  # time of event
                                            'type',  # type of event
                                            'value'  # cbg/cgm or smbg value
                                            ])

        raw_time = read_data_raw['time'].values
        raw_type = read_data_raw['type'].values
        raw_value = read_data_raw['value'].values
        raw_carbInput = read_data_raw['carbInput'].values
        raw_insulinCarbRatio = read_data_raw['insulinCarbRatio'].values
        raw_insulinOnBoard = read_data_raw['insulinOnBoard'].values
        raw_normal = read_data_raw['normal'].values
        raw_deliveryType = read_data_raw['deliveryType'].values
        raw_rate = read_data_raw['rate'].values


        # process cbg inputs first (all other variables will only be added if the are in the time-span, where cbg values were collected
        cbg_time = []

        cbg_value = []
        for i in range(len(raw_time)):
            if raw_type[i] == 'cbg':
                cbg_time.append(int(pd.to_datetime(raw_time[i]).timestamp() / 300))  # divide time by 300 to get 5 minute intervals
                cbg_value.append(float(raw_value[i]))
        cbg_time = np.array(cbg_time)
        cbg_value = np.array(cbg_value)
        sorter = np.argsort(cbg_time)
        cbg_time = cbg_time[sorter]
        cbg_value = cbg_value[sorter]
        # get a continuous time-scale from cbg_time-start to cbg_time-end with 5 minutes intervals
        basetime = np.linspace(cbg_time.copy()[0], cbg_time.copy()[-1] + 1, cbg_time.copy()[-1] + 1 - cbg_time.copy()[0])
        dict['5minute_intervals_timestamp'] = basetime  # basetime will be used to index our dataframe

        time = []
        for t in basetime:
            time.append(pd.to_datetime(int(t) * 300, unit='s'))
        dict['time'] = time
        zerotime = cbg_time.copy()[0]
        # do interpolation
        cbg_time = np.array(cbg_time) - zerotime
        out = np.full(len(basetime), np.nan)

        for i, t in enumerate(cbg_time):
            if int(cbg_time[i]) < len(basetime) and int(cbg_time[i]) >= 0:  # check for "int(time[i]) >= 0" since timestamps from other parameters prior to the first timestamp from glucose_values are igrnored
                out[int(cbg_time[i])] = cbg_value[i]
        dict['cbg'] = out*18

        #Removed interpolation using SARIMAX
        #_______________________________________
        #for i, t in enumerate(cbg_time):
        #    if int(cbg_time[i]) < len(basetime) and int(cbg_time[i]) >= 0:  # check for "int(time[i]) >= 0" since timestamps from other parameters prior to the first timestamp from glucose_values are igrnored
        #        out[int(cbg_time[i])] = cbg_value[i]
        #dict['cbg_raw'] = out

        #model = SARIMAX(out, order=(1, 1, 1))
        #results = model.fit()
        #out_imputed = results.predict()
        #out_imputed[0] = out[0]
        #dict['cbg'] = out_imputed
        #_______________________________________

        # mask for missing cbg data
        missing_cbg =(np.isnan(out)).astype(float)
        dict['missing_cbg'] = missing_cbg

        # add basal-rate
        basal_eventtime = []  # a basal event can either be a suspend event or a schedule/re-schedule event
        basal_value = []
        for i in range(len(raw_time)):
            if raw_type[i] == 'basal':
                if raw_deliveryType[i] == 'scheduled' or raw_deliveryType[i] == 'temp' or raw_deliveryType[i] == 'automated':
                    if not np.isnan(raw_rate[i]):
                        basal_value.append(float(raw_rate[i]))
                        basal_eventtime.append(pd.to_datetime(raw_time[i]).timestamp() / 300)  # 5min a 60 sek = 300
                    else:
                        print('rate is nan for "{}" in {} at line {}'.format(raw_deliveryType[i], patient_id, i))
                        # raise ValueError('rate is nan for "{}"'.format(raw_deliveryType[i]))
                elif raw_deliveryType[i] == 'suspend':
                    basal_value.append(0.0)
                    basal_eventtime.append(pd.to_datetime(raw_time[i]).timestamp() / 300)  # 5min a 60 sek = 300
                else:
                    raise ValueError('invalid deliveryType: {} in {}'.format(raw_deliveryType[i], patient_id))
        basal_eventtime = np.array(basal_eventtime)
        basal_value = np.array(basal_value)
        sorter = np.argsort(basal_eventtime)
        basal_eventtime = basal_eventtime[sorter]
        basal_value = basal_value[sorter]
        # do interpolation
        basal_eventtime = np.array(basal_eventtime) - zerotime
        out = np.full(len(basetime), np.nan)
        # for basal rates, use forward imputation
        for i in range(len(basal_eventtime)):
            if int(basal_eventtime[i]) < len(basetime):
                out[int(basal_eventtime[i]):] = basal_value[i]
        dict['basal'] = out

        # process bolus data
        bolus_time = []
        bolus_value = []
        for i in range(len(raw_time)):
            if raw_type[i] == 'bolus':
                bolus_time.append(pd.to_datetime(raw_time[i]).timestamp() / 300)  # divide time by 300 to get 5 minute intervals
                bolus_value.append(float(raw_normal[i]))
        bolus_time = np.array(bolus_time)
        bolus_value = np.array(bolus_value)
        sorter = np.argsort(bolus_time)
        bolus_time = bolus_time[sorter]
        bolus_value = bolus_value[sorter]
        # do interpolation
        bolus_time = np.array(bolus_time) - zerotime
        out = np.full(len(basetime), np.nan)
        for i in range(len(bolus_time)):
            if int(bolus_time[i]) < len(basetime) and int(bolus_time[i]) >= 0:  # check for "int(time[i]) >= 0" since timestamps from other parameters prior to the first timestamp from glucose_values are igrnored
                out[int(bolus_time[i])] = bolus_value[i]
        dict['bolus'] = out

        # process carbInput
        carbInput_time = []
        carbInput_value = []
        for i in range(len(raw_time)):
            if not np.isnan(raw_carbInput[i]):
                carbInput_time.append(pd.to_datetime(raw_time[i]).timestamp() / 300)  # divide time by 300 to get 5 minute intervals
                carbInput_value.append(float(raw_carbInput[i]))
        carbInput_time = np.array(carbInput_time)
        carbInput_value = np.array(carbInput_value)
        sorter = np.argsort(carbInput_time)
        carbInput_time = carbInput_time[sorter]
        carbInput_value = carbInput_value[sorter]
        # do interpolation
        carbInput_time = np.array(carbInput_time) - zerotime
        out = np.full(len(basetime), np.nan)
        for i in range(len(carbInput_time)):
            if int(carbInput_time[i]) < len(basetime) and int(carbInput_time[i]) >= 0:  # check for "int(time[i]) >= 0" since timestamps from other parameters prior to the first timestamp from glucose_values are igrnored
                out[int(carbInput_time[i])] = carbInput_value[i]
        dict['carbInput'] = out

        # process insulinCarbRatio
        insulinCarbRatio_time = []
        insulinCarbRatio_value = []
        for i in range(len(raw_time)):
            if not np.isnan(raw_insulinCarbRatio[i]):
                insulinCarbRatio_time.append(pd.to_datetime(raw_time[i]).timestamp() / 300)  # divide time by 300 to get 5 minute intervals
                insulinCarbRatio_value.append(float(raw_insulinCarbRatio[i]))
        insulinCarbRatio_time = np.array(insulinCarbRatio_time)
        insulinCarbRatio_value = np.array(insulinCarbRatio_value)
        sorter = np.argsort(insulinCarbRatio_time)
        insulinCarbRatio_time = insulinCarbRatio_time[sorter]
        insulinCarbRatio_value = insulinCarbRatio_value[sorter]
        # do interpolation
        insulinCarbRatio_time = np.array(insulinCarbRatio_time) - zerotime
        out = np.full(len(basetime), np.nan)
        for i in range(len(insulinCarbRatio_time)):
            if int(insulinCarbRatio_time[i]) < len(basetime) and int(insulinCarbRatio_time[i]) >= 0:  # check for "int(time[i]) >= 0" since timestamps from other parameters prior to the first timestamp from glucose_values are igrnored
                out[int(insulinCarbRatio_time[i])] = insulinCarbRatio_value[i]
        dict['insulinCarbRatio'] = out

        # process insulinOnBoard
        insulinOnBoard_time = []
        insulinOnBoard_value = []
        for i in range(len(raw_time)):
            if not np.isnan(raw_insulinOnBoard[i]):
                insulinOnBoard_time.append(pd.to_datetime(raw_time[i]).timestamp() / 300)  # divide time by 300 to get 5 minute intervals
                insulinOnBoard_value.append(float(raw_insulinOnBoard[i]))
        insulinOnBoard_time = np.array(insulinOnBoard_time)
        insulinOnBoard_value = np.array(insulinOnBoard_value)
        sorter = np.argsort(insulinOnBoard_time)
        insulinOnBoard_time = insulinOnBoard_time[sorter]
        insulinOnBoard_value = insulinOnBoard_value[sorter]
        # do interpolation
        insulinOnBoard_time = np.array(insulinOnBoard_time) - zerotime
        out = np.full(len(basetime), np.nan)
        for i in range(len(insulinOnBoard_time)):
            if int(insulinOnBoard_time[i]) < len(basetime) and int(insulinOnBoard_time[i]) >= 0:  # check for "int(time[i]) >= 0" since timestamps from other parameters prior to the first timestamp from glucose_values are igrnored
                out[int(insulinOnBoard_time[i])] = insulinOnBoard_value[i]
        dict['insulinOnBoard'] = out

        # save data frame
        df = pd.DataFrame(dict)
        df.set_index('5minute_intervals_timestamp')

        #df.to_csv(path_or_buf='../SAP100_processed_time/{}/{}_processed.csv'.format(partition, patient_id), index=False)
        #print(df.head(50))
        
        #df.to_csv(path_or_buf='mhm/{}_processed.csv'.format(patient_id), index=False)
        if calculate_iob == True:
            df['iob'] = calculate_total_iob(df['bolus'].values, ts_min=5, t_action_max_min=240)
            df['iob'] = pd.Series(df['iob']).rolling(window=12, min_periods=1).mean().to_numpy()
            df['cob'] = calculate_total_cob(df['carbInput'].values, carb_absorption=0.8, ts_min=5, t_action_max_min=240)
            df['cob'] = pd.Series(df['cob']).rolling(window=12, min_periods=1).mean().to_numpy()
        return df, patient_id


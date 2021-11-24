# modules
import pandas as pd
import cgmfeature as cgm #use this way of importing because some function have the same name as sns
import sensorsfeature as sns
import loocvRF as lor

# 1 - load data. There are 8 different datasets
validPids = ['002', '004', '005', '006', '007', '008'] # 003 e 009 might have problems with data
for pid in validPids:
    pass

pid = '002'
df = pd.read_csv(f'/Volumes/TOSHIBA EXT/D1NAMO/diabetes_subset/{pid}/{pid}_cgm_acc')

# store the result in a dictionary { key = subject number, value = loaded df }

# 2 - for every dataset, i need to chack that
# the dt column is in datetime format,
# the PeakAccel data has values that are not grater than 1, substitute nan for invalid values
# glucose values need to be converted(inplace) as: new_values = old_value * 18
# column dt must be renamed in 'datetime'

# 3 - creating functions for generating x variables - already done

def sensordata(df, feature):
    """
    df is the raw data
    feature: feature name to extract - string
    """
    df['Time'] = df['datetime']
    df['Day'] = df['Time'].dt.date

    df_sensor = df[['Time', 'Day', feature]]
    df_sensor.columns = ['Time', 'Day', 'Sensor']
    df_sensor = df_sensor[df_sensor['Sensor'].notna()]

    return df_sensor


def xs_vars(df):
    """
    df is a dataframe with time date and sensor columns, obtained from sensordata(df, feature)
    """
    interdaycv = sns.interdaycv(df)
    interdaysd = sns.interdaysd(df)
    intradaycv_mean, intradaycv_median, intradaycv_sd = sns.intradaycv(df)
    intradaysd_mean, intradaysd_median, intradaysd_sd = sns.intradaysd(df)
    intradaymean_mean, intradaymean_median, intradaymean_sd = sns.intradaymean(df)
    stir = sns.TIR(df, sd=1, sr=1)
    stor = sns.TOR(df, sd=1, sr=1)
    spor = sns.POR(df, sd=1, sr=1)
    mase = sns.MASE(df, sd=1)
    interdaymean, interdaymedian, interdaymin, interdaymax, interdayQ1, interdayQ3 = sns.summarymetrics(df)

    xrow = [interdaycv,
            interdaysd,
            intradaycv_mean, intradaycv_median, intradaycv_sd,
            intradaysd_mean, intradaysd_median, intradaysd_sd,
            intradaymean_mean, intradaymean_median, intradaymean_sd,
            stir, stor, spor, mase.at['Sensor'],
            interdaymean, interdaymedian, interdaymin, interdaymax, interdayQ1, interdayQ3]

    return xrow

# 4 - creating functions for y variable

def glucodata(df):
    """
        mod df for calculating metrics of interest
        Args:
            df
        Returns:
            (pd.DataFrame): dataframe of data with Time, Glucose, and Day columns
    """

    df['Time'] = df['datetime']
    # df.drop(df.index[:12], inplace=True)
    # df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%S')
    df['Day'] = df['Time'].dt.date
    df['Glucose'] = df['glucose']
    df = df[['Day', 'Time', 'Glucose']]
    df = df[df['Glucose'].notna()]
    # df = df.reset_index()
    return df


def get_ys_vars_dict(df):
    """
    df with time day and glucose var
    """
    ypir = cgm.PIR(df)  # in %

    yrow_vals = [ypir]

    yrow_keys = ['ypir']

    return dict(zip(yrow_keys, yrow_vals))

# 5 - create a function that takes the data, the variable to use as x and the subject id (pid) and creates a row with
# X's and the y variable. - Already done

def get_final_row2(df, pid: str, var_used_4_current_hyp: list, yname: str):
    """
    df is row data
    pid is a string
    """
    pid = int(pid)
    var = list(set(var_used_4_current_hyp) - set(["datetime", 'glucose']))

    final_row = []
    final_row.append(pid)

    for feature in var:
        df_sensor = sensordata(df, feature)
        final_row = final_row + xs_vars(df_sensor)

    glu_data = glucodata(df)
    ys = get_ys_vars_dict(glu_data).get(yname)

    final_row.append(ys)

    return final_row

# 6 - create a process for getting multiple rows in the final dataset based on resampling frequencies and shifting of
# y variable - already done

# settings
subset = ['PeakAccel']
var_used_4_current_hyp = ["datetime",'glucose'] + subset
grouping_freq = '2D'
yname = 'ypir'
shift = 0 # shift = 1 is one day in the future

# process
rowsConteiner = []
for pid in validPids:
    df = tr.get(pid)
    # groupby your key and freq
    g = df.groupby(pd.Grouper(key='datetime', freq=grouping_freq))
    # groups to a list of dataframes with list comprehension
    dfs = [group for _,group in g]
    for periodo in dfs:
        row = get_final_row2(periodo, pid, var_used_4_current_hyp, yname)
        rowsConteiner.append(row)

numOfXs = len(list(set(var_used_4_current_hyp)-set(["datetime", 'glucose'])))
x_name_list = ['x' + str(x) for x in range(1,(21*numOfXs)+1)]
colNames = ['pid'] + x_name_list + ['y']

pd2 = pd.DataFrame(rowsConteiner, columns = colNames)
pd3 = pd.DataFrame( columns = colNames)
pd2ByPID = dict(list(pd2.groupby(['pid'])))
for single_df in pd2ByPID.values(): # intermediate df for shifting
    single_df['y'] = single_df['y'].shift(-shift)
    pd3 = pd.concat([pd3, single_df])
pd3.dropna(inplace = True)

# 7 - trainig - already done

errors, meanrmse, stdrmse, meanmape, stdmape, __ = lor.loocvRF2(data=pd3, idcolumn='pid', outcomevar='y',
                                                                numestimators=1000, fs=0.005)
metrics = [meanrmse, stdrmse, meanmape, stdmape]
info = subset + [grouping_freq] + [yname] + [shift]

res = pd.DataFrame([metrics + info],
                   columns=['Mean RMSE', 'Std RMSE', 'Mean MAPE', 'Std MAPE', 'var_used', 'grouping_freq', 'yname',
                            'shift'])
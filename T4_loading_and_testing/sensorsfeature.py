    # -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:14:51 2021

@author: tobg
"""

import numpy as np
import pandas as pd
import datetime as datetime

'''
    Compute Wearable Variability Metrics:
    This algorithm contains 10 functions with 21 features for feature engineering of wearables data. It contains 2 functions to import and format wearables data. 
    Input:
        If using our import functions: filename (string): filename to .csv with Datetime in first column (format: format='%Y-%m-%d %H:%M:%S.%f') and the sensor data in the second column. Use E4FileFormatter.ipynb at dbdp.org for Empatica E4 files.
        If using functions only: dataframe with three columns: Time, Sensor, Day **For TOR, TIR, POR, sampling rate is required.
    Metrics computed:
        Interday Mean 
        Interday Median 
        Interday Maximum 
        Interday Minimum 
        Interday Standard Deviation 
        Interday Coefficient of Variation 
        Intraday Standard Deviation (mean, median, standard deviation)
        Intraday Coefficient of Variation (mean, median, standard deviation)
        Intraday Mean (mean, median, standard deviation)
        TIR (Time in Range of default 1 SD)
        TOR (Time outside Range of default 1 SD)
        POR (Percent outside Range of default 1 SD)
        MASE (Mean Amplitude of Sensor Excursions, default 1 SD)
        Q1G (intraday first quartile glucose)
        Q3G (intraday third quartile glucose)
        ** for more information on these variability metrics see dbdp.org**
        
    '''


def importe4(filename, f='%Y-%m-%d %H:%M:%S.%f'):
    """
        Function for importing and formatting for use with other functions.
        Args:
            filename (string): filename to a .csv with 2 columns - one with time in format = '%Y-%m-%d %H:%M:%S.%f', and the other column being the sensor value
            f (string): Datetime format of .csv
        Returns:
            df (pandas.DataFrame): 
    """
    df = pd.read_csv(filename, header=None, names=['Time', 'Sensor']) 
    df['Time'] =  pd.to_datetime(df['Time'], format=f)
    df['Day'] = df['Time'].dt.date
    df = df.reset_index()
    return df

def importe4acc(filename, f='%Y-%m-%d %H:%M:%S.%f'):
    """
        Function for importing and formatting for use with other functions.
        Args:
            filename (string): filename to a .csv with 4 columns - one with time in format = '%Y-%m-%d %H:%M:%S.%f', and the other columns being x,y,z of tri-axial accelerometry
            f (string): Datetime format of .csv
        Returns:
            df (pandas.DataFrame): 
    """
    df = pd.read_csv(filename, header=None, names=['Time', 'X', 'Y', 'Z']) 
    df['Time'] =  pd.to_datetime(df['Time'], format=f)
    df['Day'] = df['Time'].dt.date
    df['ri'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    df['Sensor'] = df['ri'] - 64
    df = df.drop(columns=['ri', 'X', 'Y', 'Z'])
    df = df.reset_index()
    return df


def interdaycv(df):
    """
        computes the interday coefficient of variation on pandas dataframe Sensor column
        Args:
            df (pandas.DataFrame):
        Returns:
            cvx (IntegerType): interday coefficient of variation 
    """
    try:
        cvx = (np.std(df['Sensor']) / (np.nanmean(df['Sensor'])))*100
        return cvx
    except:
        return float('nan')

def interdaysd(df):
    """
        computes the interday standard deviation of pandas dataframe Sensor column
        Args:
             df (pandas.DataFrame):
        Returns:
            interdaysd (IntegerType): interday standard deviation 
    """
    try:
        interdaysd = np.std(df['Sensor'])
        return interdaysd
    except:
        return float('nan')

def intradaycv(df):
    """
        computes the intradaycv, returns the mean, median, and sd of intraday cv Sensor column in pandas dataframe
        Args:
             df (pandas.DataFrame):
        Returns:
            intradaycv_mean (IntegerType): Mean, Median, and SD of intraday coefficient of variation 
            intradaycv_median (IntegerType): Median of intraday coefficient of variation 
            intradaycv_sd (IntegerType): SD of intraday coefficient of variation 
    """
    intradaycv = []
    
    for i in pd.unique(df['Day']):
        intradaycv.append(interdaycv(df[df['Day']==i]))
    
    try:
        intradaycv_mean = np.nanmean(intradaycv)
    except:
        intradaycv_mean = float('nan')
    
    try:
        intradaycv_median = np.nanmedian(intradaycv)
    except:
        intradaycv_median = float('nan')
    
    
    try:
        intradaycv_sd = np.nanstd(intradaycv)
    except:
        intradaycv_sd = float('nan')
    
    return intradaycv_mean, intradaycv_median, intradaycv_sd

def intradaysd(df):
    """
        computes the intradaysd, returns the mean, median, and sd of intraday sd Sensor column in pandas dataframe
        Args:
             df (pandas.DataFrame):
        Returns:
            intradaysd_mean (IntegerType): Mean, Median, and SD of intraday standard deviation 
            intradaysd_median (IntegerType): Median of intraday standard deviation 
            intradaysd_sd (IntegerType): SD of intraday standard deviation 
    """
    intradaysd =[]
    for i in pd.unique(df['Day']):
        intradaysd.append(np.std(df[df['Day']==i].drop('Time', axis = 1)))
    
    try:
        intradaysd_mean = np.nanmean(intradaysd)
    except:
        intradaysd_mean = float('nan')
    
    try:
        intradaysd_median = np.nanmedian(intradaysd)
    except:
        intradaysd_median = float('nan')
    
    try:
        intradaysd_sd = np.nanstd(intradaysd)
    except:
        intradaysd_sd = float('nan')
    
    return intradaysd_mean, intradaysd_median, intradaysd_sd

def intradaymean(df):
    """
        computes the intradaymean, returns the mean, median, and sd of the intraday mean of the Sensor data
        Args:
             df (pandas.DataFrame):
        Returns:
            intradaysd_mean (IntegerType): Mean, Median, and SD of intraday standard deviation of glucose
            intradaysd_median (IntegerType): Median of intraday standard deviation of glucose
            intradaysd_sd (IntegerType): SD of intraday standard deviation of glucose
    """
    intradaymean =[]
    for i in pd.unique(df['Day']):
        intradaymean.append(np.mean(df[df['Day']==i]))
    
    try:
        intradaymean_mean = np.mean(intradaymean)
    except:
        intradaymean_mean = float('nan')
    try:
        intradaymean_median = np.median(intradaymean)
    except:
        intradaymean_median = float('nan')
    try:
        intradaymean_sd = np.std(intradaymean)
    except:
        intradaymean_sd = float('nan')
        
    return intradaymean_mean, intradaymean_median, intradaymean_sd


def TIR(df, sd=1, sr=1):
    """
        computes time in the range of (default=1 sd from the mean) sensor column in pandas dataframe
        Args:
             df (pandas.DataFrame):
             sd (IntegerType): standard deviation from mean for range calculation (default = 1 SD)
             sr (IntegerType): 
        Returns:
            TIR (IntegerType): Time in Range set by sd, *Note time is relative to your SR
            
    """
    
    try:
        up = np.mean(df['Sensor']) + sd*np.std(df['Sensor'])
        dw = np.mean(df['Sensor']) - sd*np.std(df['Sensor'])
        TIR = len(df[(df['Sensor']<= up) & (df['Sensor']>= dw)])*sr 
        return TIR
    except:
        return float('nan')



def TOR(df, sd=1, sr=1):
    """
        computes time outside the range of (default=1 sd from the mean) glucose column in pandas dataframe
        Args:
             df (pandas.DataFrame):
             sd (IntegerType): standard deviation from mean for range calculation (default = 1 SD)
             sr (IntegerType): 
        Returns:
            TOR (IntegerType): Time outside of range set by sd, *Note time is relative to your SR
    """
    try:
        up = np.mean(df['Sensor']) + sd*np.std(df['Sensor'])
        dw = np.mean(df['Sensor']) - sd*np.std(df['Sensor'])
        TOR = len(df[(df['Sensor']>= up) | (df['Sensor']<= dw)])*sr
        return TOR
    except:
        return float('nan')


def POR(df, sd=1, sr=1):
    """
        computes percent time outside the range of (default=1 sd from the mean) sensor column in pandas dataframe
        Args:
             df (pandas.DataFrame):
             sd (IntegerType): standard deviation from mean for range calculation (default = 1 SD)
             sr (IntegerType): 
        Returns:
            POR (IntegerType): percent of time spent outside range set by sd
    """
    try:
        up = np.mean(df['Sensor']) + sd*np.std(df['Sensor'])
        dw = np.mean(df['Sensor']) - sd*np.std(df['Sensor'])
        TOR = len(df[(df['Sensor']>= up) | (df['Sensor']<= dw)])*sr
        POR = (TOR/(len(df)*sr))*100
        return POR
    except:
        return float('nan')


def MASE(df, sd=1):
    """
        computes the mean amplitude of sensor excursions (default = 1 sd from the mean)
        Args:
             df (pandas.DataFrame):
             sd (IntegerType): standard deviation from mean to set as a sensor excursion (default = 1 SD)
        Returns:
           MASE (IntegerType): Mean Amplitude of sensor excursions
    """
    try:
        up = np.mean(df['Sensor']) + sd*np.std(df['Sensor'])
        dw = np.mean(df['Sensor']) - sd*np.std(df['Sensor'])
        MASE = np.mean(df[(df['Sensor']>= up) | (df['Sensor']<= dw)])
        return MASE
    except:
        return float('nan')



def summarymetrics(df):
    """
        computes interday mean, median, minimum and maximum, and first and third quartile 
        Args:
             df (pandas.DataFrame):
        Returns:
            interdaymean (FloatType): mean 
            interdaymedian (FloatType): median 
            interdaymin (FloatType): minimum 
            interdaymax (FloatType): maximum 
            interdayQ1 (FloatType): first quartile 
            interdayQ3 (FloatType): third quartile 
    """

    try:
        interdaymean = np.nanmean(df['Sensor'])
    except:
        interdaymean = float('nan')

    try:
        interdaymedian = np.nanmedian(df['Sensor'])
    except:
        interdaymedian = float('nan')

    try:
        interdaymin = np.nanmin(df['Sensor'])
    except:
        interdaymin = float('nan')

    try:
        interdaymax = np.nanmax(df['Sensor'])
    except:
        interdaymax = float('nan')

    try:
        interdayQ1 = np.nanpercentile(df['Sensor'], 25)
    except:
        interdayQ1 = float('nan')

    try:
        interdayQ3 = np.nanpercentile(df['Sensor'], 75)
    except:
        interdayQ3 = float('nan')


    return interdaymean, interdaymedian, interdaymin, interdaymax, interdayQ1, interdayQ3
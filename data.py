#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os
import sys
import inspect
import types
import numpy as np
import pandas as pd

from nystrom import exact_kernel

DEFAULT_3D_SPATIAL_NETWORK_PATH = os.path.join(
    os.getcwd(), "data", "3D_spatial_network.csv")
DEFAULT_ABALONE_PATH = os.path.join(
    os.getcwd(), "data", "abalone.csv")
DEFAULT_MAGIC04_PATH = os.path.join(
    os.getcwd(), "data", "magic04.csv")
DEFAULT_SLICE_LOCALIZATION_PATH = os.path.join(
    os.getcwd(), "data", "slice_localization_data.csv")
DEFAULT_WINE_PATH = os.path.join(
    os.getcwd(), "data", "Wine_data.csv")
DEFAULT_GAS_SENSOR_PATH = os.path.join(
    os.getcwd(), "data", "gas_sensor", "ethylene_CO.csv")
DEFAULT_TEMPERATURE_DATA_PATH = os.path.join(
    os.getcwd(), "data", "nearest_rainfall_SILO", "code", "vic120", "data", "stations_1990_2020", "station_id_78020.csv")
DEFAULT_STOCK_MARKET_PATH = os.path.join(
    os.getcwd(), "data", "stock_market.csv")
DEFAULT_SPAM_PATH = os.path.join(
    os.getcwd(), "data", "spambase.data")

"""
Potential candidates:
https://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures
https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
https://archive.ics.uci.edu/ml/datasets/Wave+Energy+Converters
https://archive.ics.uci.edu/ml/datasets/Air+Quality
"""


def is_loader(object):
    return isinstance(object, types.FunctionType) and (object.__module__ == __name__) and ("load_" in str(object))


def load_3D_spatial_network(path: str = DEFAULT_3D_SPATIAL_NETWORK_PATH, labels: bool = False):
    """
    REGRESSION

    Loads the 3D Road Network (North Jutland, Denmark) Data Set.

    Source: https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2C+Denmark%29
    Number of Instances: 434874
    Number of Attributes: 4
    Attribute Information (Raw):
        1. OSM_ID: OpenStreetMap ID for each road segment or edge in the graph.
        2. LONGITUDE: Web Mercaptor (Google format) longitude
        3. LATITUDE: Web Mercaptor (Google format) latitude
        4. ALTITUDE: Height in meters.
    Description:
    This dataset was constructed by adding elevation information to a 2D 
    road network in North Jutland, Denmark (covering a region of 185 x 135 km^2).
    Additional Notes:
    The loaded data set will have the OSM_ID removed to better suit the RBF 
    kernel. The labels are the ALTITUDE values. The number of samples is 
    capped at 40000 for memory.
    """
    headers = [
        "OSM_ID",
        "LONGITUDE",
        "LATITUDE",
        "ALTITUDE",
    ]
    df = pd.read_csv(path, names=headers)
    df.drop(columns=["OSM_ID"], inplace=True)
    data = df[["LONGITUDE", "LATITUDE"]]
    data = data.to_numpy(dtype=float)[:20_000, :]

    if labels:
        return data, df["ALTITUDE"].to_numpy(dtype=float)[:20_000]

    return data


def load_abalone(path: str = DEFAULT_ABALONE_PATH, labels: bool = False):
    """
    CLASSIFICATION (Intended)
    REGRESSION (Shell weight)

    Loads the abalone Data Set.

    Source: https://archive.ics.uci.edu/ml/datasets/abalone
    Number of Instances: 4177
    Number of Attributes: 8
    Attribute Information (Raw):
        1. Sex / nominal / -- / M, F, and I (infant)
        2. Length / continuous / mm / Longest shell measurement
        3. Diameter / continuous / mm / perpendicular to length
        4. Height / continuous / mm / with meat in shell
        5. Whole weight / continuous / grams / whole abalone
        6. Shucked weight / continuous / grams / weight of meat
        7. Viscera weight / continuous / grams / gut weight (after bleeding)
        8. Shell weight / continuous / grams / after being dried
        9. Rings / integer / -- / +1.5 gives the age in years 
    Description:
    Predicting the age of abalone from physical measurements. 
    The age of abalone is determined by cutting the shell through the cone, 
    staining it, and counting the number of rings through a microscope -- 
    a boring and time-consuming task. Other measurements, which are easier 
    to obtain, are used to predict the age.
    Additional Notes:
    The loaded data set will have the Sex removed to better suit the RBF 
    kernel. Shell weight will be removed to become part of the test labelling.
    """
    headers = "Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings".split(
        sep=",")
    df = pd.read_csv(path, names=headers)
    shell_weight = df["Shell weight"]
    df.drop(columns=["Sex", "Shell weight"], inplace=True)
    data = df.to_numpy(dtype=float)

    if labels:
        return data, shell_weight.to_numpy(dtype=float)

    return data


def load_magic04(path: str = DEFAULT_MAGIC04_PATH, labels: bool = False):
    """
    CLASSIFICATION

    Loads the magic04 Data Set.

    Source: https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope
    Number of Instances: 19020
    Number of Attributes: 11
    Attribute Information (Raw):
        1.  fLength:  continuous  # major axis of ellipse [mm]
        2.  fWidth:   continuous  # minor axis of ellipse [mm] 
        3.  fSize:    continuous  # 10-log of sum of content of all pixels [in #phot]
        4.  fConc:    continuous  # ratio of sum of two highest pixels over fSize  [ratio]
        5.  fConc1:   continuous  # ratio of highest pixel over fSize  [ratio]
        6.  fAsym:    continuous  # distance from highest pixel to center, projected onto major axis [mm]
        7.  fM3Long:  continuous  # 3rd root of third moment along major axis  [mm] 
        8.  fM3Trans: continuous  # 3rd root of third moment along minor axis  [mm]
        9.  fAlpha:   continuous  # angle of major axis with vector to origin [deg]
        10.  fDist:    continuous  # distance from origin to center of ellipse [mm]
        11.  class:    g,h         # gamma (signal), hadron (background)
    Description:
    The data are MC generated (see below) to simulate registration of high energy
    gamma particles in a ground-based atmospheric Cherenkov gamma telescope using the
    imaging technique.

    g = gamma (signal):     12332
    h = hadron (background): 6688

    For technical reasons, the number of h events is underestimated.
    In the real data, the h class represents the majority of the events.
    Additional Notes:
    The labels are the fDist values.
    """
    headers = [
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
        "class",
    ]
    df = pd.read_csv(path, names=headers)
    labels_vec = df["class"].replace({'g': 1, 'h': 0}).to_numpy(dtype=int)
    df.drop(columns=["class"], inplace=True)
    data = df.to_numpy(dtype=float)

    if labels:
        return data, labels_vec

    return data


def load_Wine_data(path: str = DEFAULT_WINE_PATH, labels: bool = False):
    """
    CLASSIFICATION

    Loads the slice localization Data Set.

    Source: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    Number of Instances: 4898
    Number of Attributes: 12
    Attribute Information (Raw):
        1 - fixed acidity (continuous)
        2 - volatile acidity (continuous)
        3 - citric acid (continuous)
        4 - residual sugar (continuous)
        5 - chlorides (continuous)
        6 - free sulfur dioxide (continuous)
        7 - total sulfur dioxide (continuous)
        8 - density (continuous)
        9 - pH (continuous)
        10 - sulphates (continuous)
        11 - alcohol (%) (continuous)
        Output variable (based on sensory data):
        12 - quality (integer score between 0 and 10)
    Description:
    Two datasets are included, related to red and white vinho verde wine 
    samples, from the north of Portugal. The goal is to model wine quality 
    based on physicochemical tests.
    Additional Notes:
    The labels are the "quality" values.
    """

    df = pd.read_csv(path)
    labels_vec = df["quality"].to_numpy(dtype=int)
    df.drop(columns=["quality"], inplace=True)
    data = df.to_numpy(dtype=float)
    labels_vec[labels_vec < 7] = 0
    labels_vec[labels_vec >= 7] = 1

    if labels:
        return data, labels_vec

    return data


def load_gas_sensor_data(path: str = DEFAULT_GAS_SENSOR_PATH, labels: bool = False):
    """
    REGRESSION

    Loads the Gas Sensor Data Set.

    Source: https://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures
    Number of Instances: 4178504
    Number of Attributes: 4
    Attribute Information (Raw):
        1. Time (seconds)
        2. Methane conc (ppm)
        3. Ethylene conc (ppm)
        4-20. sensor readings (16 channels) 
    Description:
    This data set contains the acquired time series from 16 chemical sensors 
    exposed to gas mixtures at varying concentration levels.
    Additional Notes:
    Ethylene ppm will be omitted. The labels are the CO ppm values.
    """
    headers = [
        "time",
        "CO_ppm",
        "ETHYLENE_ppm",
    ] + [f"SEN_{i}" for i in range(16)]
    # skip the first line of this data file, header is not formatted properly
    df = pd.read_csv(path, names=headers, skiprows=[0], delim_whitespace=True)
    labels_vec = df["CO_ppm"].to_numpy(dtype=float)
    df.drop(columns=["CO_ppm", "ETHYLENE_ppm"], inplace=True)
    data = df.to_numpy(dtype=float)[0:1_200_000:(1_200_000//20_000), ::]

    if labels:
        return data, labels_vec[0:1_200_000:(1_200_000//20_000)]

    return data


def load_temperature_data(path: str = DEFAULT_TEMPERATURE_DATA_PATH, labels: bool = False):
    """
    REGRESSION

    Loads the temperature Data Set.

    Source: https://www.longpaddock.qld.gov.au/
    Number of Instances: ~11000
    Number of Attributes: 9
    Attribute Information (Raw):
        1. Station Id
        2. Date (YYYY-MM-DD)
        3. daily rainfall
        4. daily rain source
        5. maximum temperature
        6. maximum temperature source
        7. minimum temperature
        8. minimum temperature source
        9. meta data
    Description:
    Weather station data from rural Queensland.
    Additional Notes:
    We will only be interested in data and maximum temperature.
    """

    df = pd.read_csv(path)

    date_format = r"%Y-%m-%d"
    times = df["YYYY-MM-DD"]
    times = pd.to_datetime(times, format=date_format)
    times = times.diff(1).dt.days
    times = times.fillna(0.0)
    times = times.astype(float)
    times = times.cumsum(axis="index", skipna=True).to_numpy(
        dtype=float).reshape(-1, 1)

    if not labels:
        return times

    return times, df["max_temp"].to_numpy(dtype=float)


def load_stock_market(path: str = DEFAULT_STOCK_MARKET_PATH, labels: bool = False):
    """
    REGRESSION

    Loads the temperature Data Set.

    Source: see https://github.com/tiskw/random-fourier-features/blob/main/dataset/stockprice/download_stockprice_zipped_csv.py
    Number of Instances: ~4904
    Number of Attributes: 4
    Attribute Information (Raw):
        1. Date
        2. Open (Daily Stock Price)
        3. High (Daily Stock Price)
        4. Low (Daily Stock Price)
        5. Close (Daily Stock Price)
        6. Volume
        7. Adj Close (Daily Stock Price)
    Description:
    Daily stock prices spanning 2000 to 2019
    Additional Notes:
    Adj Close is removed as an input component. Volume is divided by 1000.
    The labels will be daily close.
    """

    df = pd.read_csv(path)

    date_format = r"%Y-%m-%d"
    times = df["Date"]
    times = pd.to_datetime(times, format=date_format)
    times = times.diff(1).dt.days
    times = times.fillna(0.0)
    times = times.astype(float)
    df["Date"] = times.cumsum(axis="index", skipna=True).to_numpy(
        dtype=float).reshape(-1, 1)
    df["Open"] = df["Open"].map(lambda vol: vol / 1000.0)
    df["High"] = df["High"].map(lambda vol: vol / 1000.0)
    df["Low"] = df["Low"].map(lambda vol: vol / 1000.0)
    df["Volume"] = df["Volume"].map(lambda vol: vol / 100000.0)
    df["Close"] = df["Close"].map(lambda vol: vol / 1000.0)
    close = df["Close"].to_numpy(dtype=float)
    df.drop(columns=["Adj Close", "Close"], inplace=True)
    data = df.to_numpy(dtype=float)

    if not labels:
        return data

    return data, close


def load_iris(labels: bool = False):
    """
    CLASSIFICATION

    Loads the iris data set from sklearn.

    Source: see https://github.com/tiskw/random-fourier-features/blob/main/dataset/stockprice/download_stockprice_zipped_csv.py
    Number of Instances: 150
    Number of Attributes: 4
    Attribute Information (Raw):
        1. Sepal Length
        2. Sepal Width
        3. Petal Length
        4. Petal Width
        5. Iris Type
    Description:
    This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal 
    length, stored in a 150x4 numpy.ndarray.
    """
    from sklearn import datasets
    iris = datasets.load_iris()

    data = iris.data
    if not labels:
        return data

    labels = (iris.target).reshape(-1, 1)

    return data, labels


def load_rastrigin(labels: bool = False, n=750, sigma=1.0, var=1e-5):
    """
    REGRESSION

    Loads the iris data set from sklearn.

    Number of Instances: User defined.
    Number of Attributes: 2
    Attribute Information (Raw):
        x and y positions.
    Description:
    Provides a Gaussian process of the rastrigin function over a 
    [-4,4] x [-4,4] input space.
    """
    inputs = np.random.random(size=(n, 2))
    inputs = 8*inputs - 4
    if not labels:
        return inputs
    X, Y = inputs[:, 0].squeeze(), inputs[:, 1].squeeze()
    true_mean = ((X**2) + (Y**2))
    kernel = exact_kernel(inputs, sigma=sigma) + var
    Y_G = np.random.multivariate_normal(
        true_mean, kernel)
    return inputs, Y_G


def load_spam(path: str = DEFAULT_SPAM_PATH, labels: bool = False):
    """
    CLASSIFICATION

    Loads a STANDARDIZED VERSION of the spam data set.

    Source: https://archive.ics.uci.edu/ml/datasets/spambase
    Number of Instances: 4601
    Number of Attributes: 57
    Attribute Information (Raw):
        - 48 continuous real [0,100] attributes of type word_freq_WORD
        = percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.
        - 6 continuous real [0,100] attributes of type char_freq_CHAR]
        = percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail
        - 1 continuous real [1,...] attribute of type capital_run_length_average
        = average length of uninterrupted sequences of capital letters
        - 1 continuous integer [1,...] attribute of type capital_run_length_longest
        = length of longest uninterrupted sequence of capital letters
        - 1 continuous integer [1,...] attribute of type capital_run_length_total
        = sum of length of uninterrupted sequences of capital letters
        = total number of capital letters in the e-mail
        - 1 nominal {0,1} class attribute of type spam
        = denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.

        - labels denote whether the e-mail was considered spam (1) or not (0)
    """
    from sklearn import preprocessing
    data = np.loadtxt(path, delimiter=",", skiprows=0)
    data_pro = data[:, 0:-1].astype(float)
    scaler = preprocessing.StandardScaler().fit(data_pro)
    data_pro = scaler.transform(data_pro)
    labels_pro = data[:, -1].astype(int).squeeze()

    if not labels:
        return data_pro

    return data_pro, labels_pro


def load_data(data: str, path: str = None, labels: bool = False, **kwargs):
    """
    Loads the specified data set.

    Parameters:
        data:
            The data set to load.
        path:
            The path to the data set.
        labels:
            If true, returns the labels as a separate vector.
    """
    data = "load_" + data
    data_map = dict(inspect.getmembers(
        sys.modules[__name__], predicate=is_loader))

    if data not in data_map:
        raise ValueError("No loader for " + str(data) + " found.")

    data_loader = data_map[data]

    if path is None:
        return data_loader(labels=labels, **kwargs)

    return data_loader(path=path, labels=labels, **kwargs)


def main():

    # data = load_temperature_data(labels=False)
    # data, labels = load_data("spam", labels=True)
    # load_data("rastrigin", labels=True)
    data, labels = load_data("Wine_data", labels=True)
    # print(data)
    print(labels)
    print(sum(labels))
    print(data.shape)
    print(labels.shape)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    data, labels = load_data("rastrigin", labels=True)
    X = np.linspace(-5.12, 5.12, 100)
    Y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(X, Y)
    XD, YD = data[:, 0].squeeze(), data[:, 1].squeeze()
    # X, Y = np.meshgrid(X, Y)

    ZD = labels
    Z = ((X**2 - np.cos(2 * np.pi * X)) +
         (Y**2 - np.cos(2 * np.pi * Y)))
    # print(Z.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.nipy_spectral, linewidth=0.08,
                    antialiased=True)
    ax.scatter(XD, YD, ZD, cmap=cm.nipy_spectral)
    # plt.savefig('rastrigin_graph.png')
    plt.show()
    """


if __name__ == "__main__":
    main()

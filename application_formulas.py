import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

   
def pca_maker(data_import):

    numerical_column_list = []
    categorical_column_list = []

    for i in data_import.columns:
        if data_import[i].dtype == np.dtype("float64") or data_import[i].dtype == np.dtype("int64"):
            numerical_column_list.append(data_import[i])
        else:
            categorical_column_list.append(data_import[i])

    numerical_data = pd.concat(numerical_column_list, axis=1) #axis=1 stacks columns next to each other
    categorical_data = pd.concat(categorical_column_list, axis=1)


    numerical_data = numerical_data.apply(lambda x: x.fillna(np.mean(x)))

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(numerical_data) #converts into numpy array -2d


    pca = PCA()
    pca_data = pca.fit_transform(scaled_values)

    pca_data = pd.DataFrame(pca_data)


    new_column_names = ["PCA_" + str(i) for i in range(1, len(pca_data.columns) + 1)] #making name for columns

    list(pca_data.columns)

    #to map both new and old list
    column_mapper = dict(zip(list(pca_data.columns), new_column_names))

    pca_data.rename(columns=column_mapper, inplace=True) #inplace makes changes in real data not a copy

    output = pd.concat([data_import, pca_data], axis=1)

    return output, list(categorical_data.columns), new_column_names
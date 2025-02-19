import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from application_formulas import pca_maker
from sklearn import preprocessing

st.set_page_config(layout="wide")
scatter_columns,setting_column = st.columns((4,1))

scatter_columns.title("Multi Dimensional Analysis")

setting_column.title("Settings")

uploaded_file = setting_column.file_uploader("Choose File")

if uploaded_file is not None:
    data_import = pd.read_csv(uploaded_file)

    pca_data, cat_cols, pca_cols = pca_maker(data_import)

    categorical_variable = setting_column.selectbox("Variable Select", options=cat_cols)
    categorical_variable_2 = setting_column.selectbox("Second Variable Select", options=cat_cols)


    pca_1 = setting_column.selectbox("First Principle Componenet", options=pca_cols, index=0)
    pca_cols.remove(pca_1)
    pca_2 = setting_column.selectbox("Second Principle Componenet", options=pca_cols)


    scatter_columns.plotly_chart(px.scatter(data_frame=pca_data, x=pca_1, y=pca_2, color=categorical_variable, template="simple_white", height=800, hover_data=[categorical_variable_2]),use_container_width=True)


else:
    scatter_columns.header("please choose a file")
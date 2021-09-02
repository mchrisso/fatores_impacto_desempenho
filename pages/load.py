import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from function import *

DATA_TRAIN = None


def app():
    # Globals
    # DATA_PATH = 'data/dados.csv'

    # Upload data
    file_details = None
    uploaded_file = st.file_uploader("Faça upload do arquivo:", type=['csv'])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name,
                        "FileType": uploaded_file.type,
                        "FileSize": uploaded_file.size}

        st.write(file_details)

        st.title('Dados Discentes')

        # Load Data
        data = load_data(uploaded_file)
        st.markdown('### Raw Data')
        st.dataframe(data.head().astype('object'))

        data = transform_data(data)

        data_train = data_train_generate(data)
        st.markdown('### Transformed Data')
        st.dataframe(data_train.head().astype('object'))

        data.to_csv('data/write_data/data.csv', sep=';',
                    index=False, encoding='utf-8')
        data_train.to_csv('data/write_data/data_train.csv',
                          sep=';', index=False, encoding='utf-8')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pages import load
from function import *

def app():

    st.markdown('### Correlação dos Atributos:')

    data_train = pd.read_csv(
        'data/write_data/data_train.csv', sep=';')

    fig, ax = plt.subplots(figsize=(30,20))

    sns.set(font_scale=1.2)
    sns.heatmap(data_train.corr(), cmap='YlGnBu', annot=True, linewidths=0.2, fmt='.0g', ax=ax)
    st.write(fig)
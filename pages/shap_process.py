import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pages import load
from function import *


def app():
    # Instantiating the SHAP explainer using our trained RFC model

    data_train = pd.read_csv(
        'data/write_data/data_train.csv', sep=';')

    st.markdown('### SHAP in Streamlit')
    train_and_SHAP(data_train)

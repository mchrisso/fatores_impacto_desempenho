from multiapp import MultiApp
from pages import load, shap_process, heatmap  # import your app modules here
from function import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Inserção de Dados": load,
    "SHAP": shap_process,
    "Heatmap": heatmap,
}


def main():

    st.sidebar.markdown("## Menu")
    selection = st.sidebar.radio("", list(PAGES.keys()))

    page = PAGES[selection]
    page.app()

    st.sidebar.markdown("## Release")
    st.sidebar.info(
        "Aplicação para identificação de atributos relevantes em dados educacionais"
    )
    st.sidebar.markdown("## Versão 1.0.0")
    st.sidebar.info(
        """ Aplicação é parte do trabalho de conclusão do mestrado em Ciência da Computação da UFG - Fatores de impacto no desempenho acadêmico: um estudo de caso em cursos de computação. """
    )


if __name__ == "__main__":
    main()

import shap
import numpy as np
import pandas as pd
import xgboost
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


@st.cache(allow_output_mutation=True)
def load_data(path):
    data = pd.read_csv(path)
    return data


@st.cache(allow_output_mutation=True)
def transform_data(data):
    data['faixa_etaria'] =\
        pd.cut(data['idade_ingresso'],
               bins=[15, 18, 21, 24, 27, 30, 40, 200],
               labels=['Até 18',
                       'Entre 19 e 21',
                       'Entre 21 e 24',
                       'Entre 25 e 27',
                       'Entre 28 e 30',
                       'Entre 31 e 40',
                       'Maior que 41'
                       ])

    # Preenchendo os NANS com a media
    data['media_global_aluno'] =\
        data['media_global_aluno'].fillna(data.media_global_aluno.mean())

    data['acao_afirmativa_'] = data['acao_afirmativa'].copy()
    data.replace({'acao_afirmativa_': {'Ampla Concorrência': 0,
                                       'RI': 1,
                                       'RI-PPI': 1,
                                       'RI-PPI-CD': 1,
                                       'RS': 1,
                                       'RS-PPI': 1,
                                       'RS-PPI-CD': 1}}, inplace=True, regex=True)

    data.replace({'trancamentos': {1: 1,
                                   2: 1,
                                   3: 1,
                                   4: 1,
                                   5: 1,
                                   6: 1,
                                   0: 0}}, inplace=True, regex=True)

    data['acao_afirmativa_'] = data['acao_afirmativa_'].astype(int)

    return data


def data_train_generate(data):
    data_train = data[['acao_afirmativa_', 'turno', 'escola_publica', 'sexo', 'cor_raca',
                       'media_global_aluno', 'total_componentes',
                       'percentagem_frequencia', 'percentagem_aprovacao',
                       'percentagem_reprovacao', 'trancamentos', 'mobilidade',
                       'qtd_semestres', 'estado_civil', 'faixa_etaria']].copy()
    # save status
    y = data.status

    dummies = pd.get_dummies(data_train, columns=[
        "acao_afirmativa_",
        "turno",
        "escola_publica",
        "sexo",
        "cor_raca",
        "estado_civil",
        "faixa_etaria"],
        prefix=[
        "acao_afirmativa_",
        "turno",
        "escola_publica",
        "sexo",
        "cor_raca",
        "estado_civil",
        "faixa_etaria"])
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(dummies)
    X_normalized = preprocessing.normalize(scaled_train, norm='l2')

    columns = dummies.columns
    data_train = pd.DataFrame(X_normalized, columns=columns)

    data_train['status'] = 0
    data_train['status'] = y
    data_train.replace({'status': {'Excluído': 0, 'Ativo': 2,
                       'Graduado': 1}}, inplace=True, regex=True)

    data_train = data_train[(data_train["status"] == 0)
                            | (data_train["status"] == 1)]

    return data_train


def train_and_SHAP(data_train):

    st.write('Processing Model...')

    X = data_train.iloc[:, :-1].values
    Y = data_train.status.values

    kf = KFold(n_splits=4, random_state=42, shuffle=True)

    score = cross_val_score(DecisionTreeClassifier(), X,
                            Y, cv=kf, scoring="accuracy")
    print('Pontuação de cada fold: {}'.format(score))
    print('Média: {}%'.format(round(score.mean(), 2)*100))

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    clf = clf.fit(X, Y)

    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, feature_names=data_train.columns[:-1],
                   class_names=data_train.columns[-1],
                   filled=True,
                   rounded=True)

    #############################
    st.write('Avaliação por atributo')

    x = data_train.iloc[:, :-1]

    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x)

    st_shap(shap.force_plot(
        explainer.expected_value[0], shap_values[0], x), 400)

    #############################
    st.write('Sumário')

    model = xgboost.train({"learning_rate": 0.01},
                          xgboost.DMatrix(X, label=Y), 100)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    plt.figure(figsize=(7, 5))
    st.pyplot(shap.summary_plot(shap_values, x),
              bbox_inches='tight', dpi=300, pad_inches=0)

# Instale o Streamlit e as dependências
# pip install streamlit pandas scikit-learn openpyxl

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# Constantes
CAMPOS = [
    "id_paciente", "idade", "sexo", "estagio_cancer", "tipo_tratamento",
    "equipamento_utilizado", "ocorrencia_lesao", "tipo_lesao", "interrupcao_tratamento",
    "classificacao_mallampati", "trismo", "mobilidade_cervical", "data_registro"
]

ARQUIVO = "dataset_neoplasia_cervical.csv"

# Funções
def criar_dataset():
    if not os.path.exists(ARQUIVO):
        pd.DataFrame(columns=CAMPOS).to_csv(ARQUIVO, index=False)

def adicionar_registro(dados):
    df = pd.read_csv(ARQUIVO)
    df = pd.concat([df, pd.DataFrame([dados])], ignore_index=True)
    df.to_csv(ARQUIVO, index=False)

def classificar_risco():
    try:
        df = pd.read_csv(ARQUIVO)
        if len(df) < 5:
            return None, "Insira pelo menos 5 registros para simular classificação."

        df = df.fillna(0)
        df_encoded = pd.get_dummies(df.drop(columns=['id_paciente', 'data_registro', 'tipo_lesao']))
        X = df_encoded.drop(columns=['ocorrencia_lesao_Não']) if 'ocorrencia_lesao_Não' in df_encoded else df_encoded
        y = df_encoded['ocorrencia_lesao_Não'] if 'ocorrencia_lesao_Não' in df_encoded else [0]*len(X)

        modelo = RandomForestClassifier()
        modelo.fit(X, y)
        score = modelo.score(X, y)
        return score, None
    except Exception as e:
        return None, str(e)

def exportar_para_excel():
    try:
        df = pd.read_csv(ARQUIVO)
        df.to_excel("dataset_neoplasia_cervical.xlsx", index=False)
        return True
    except Exception as e:
        return False

# App Streamlit
st.set_page_config(page_title="Gerador de Dataset Clínico", layout="centered")

st.title("📋 Sistema de Cadastro de Pacientes")
st.write("Projeto de geração de dataset clínico para análise de neoplasias cervicais.")

# Menu de Navegação
aba = st.sidebar.selectbox("Menu", ["Adicionar Paciente", "Visualizar Dados", "Classificar com IA", "Exportar para Excel"])

criar_dataset()

if aba == "Adicionar Paciente":
    st.header("➕ Adicionar Novo Paciente")
    with st.form(key='form_paciente'):
        id_paciente = st.text_input("ID do Paciente")
        idade = st.number_input("Idade", min_value=0, max_value=120)
        sexo = st.selectbox("Sexo", ["Masculino", "Feminino", "Outro"])
        estagio_cancer = st.text_input("Estágio do Câncer")
        tipo_tratamento = st.text_input("Tipo de Tratamento")
        equipamento_utilizado = st.text_input("Equipamento Utilizado")
        ocorrencia_lesao = st.selectbox("Ocorrência de Lesão", ["Sim", "Não"])
        tipo_lesao = st.text_input("Tipo de Lesão")
        interrupcao_tratamento = st.selectbox("Interrupção de Tratamento", ["Sim", "Não"])
        classificacao_mallampati = st.text_input("Classificação Mallampati")
        trismo = st.selectbox("Trismo", ["Sim", "Não"])
        mobilidade_cervical = st.text_input("Mobilidade Cervical")
        
        submit_button = st.form_submit_button("Salvar Paciente")
        
        if submit_button:
            adicionar_registro({
                "id_paciente": id_paciente,
                "idade": idade,
                "sexo": sexo,
                "estagio_cancer": estagio_cancer,
                "tipo_tratamento": tipo_tratamento,
                "equipamento_utilizado": equipamento_utilizado,
                "ocorrencia_lesao": ocorrencia_lesao,
                "tipo_lesao": tipo_lesao,
                "interrupcao_tratamento": interrupcao_tratamento,
                "classificacao_mallampati": classificacao_mallampati,
                "trismo": trismo,
                "mobilidade_cervical": mobilidade_cervical,
                "data_registro": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            st.success("Paciente cadastrado com sucesso!")

elif aba == "Visualizar Dados":
    st.header("📄 Dados dos Pacientes")
    df = pd.read_csv(ARQUIVO)
    st.dataframe(df)

elif aba == "Classificar com IA":
    st.header("🧠 Classificação de Risco (Simulação de IA)")
    score, erro = classificar_risco()
    if erro:
        st.warning(erro)
    else:
        st.success(f"Acurácia simulada do modelo: {score*100:.2f}%")

elif aba == "Exportar para Excel":
    st.header("📁 Exportação de Dados")
    sucesso = exportar_para_excel()
    if sucesso:
        st.success("Arquivo Excel gerado com sucesso: `dataset_neoplasia_cervical.xlsx`")
    else:
        st.error("Erro ao exportar para Excel.")

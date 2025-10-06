import pandas as pd
import numpy as np
import os # Para verificar a existência do arquivo

# --- Definição dos Arquivos ---
ARQUIVO_ENTRADA = "/workspaces/python/dataset_alpaca_50k/trn.json"
ARQUIVO_SAIDA_COMPLETA = "dados_limpos.json"
ARQUIVO_AMOSTRA_TREINO = "amostra_treino.json"

def limpar_e_amostrar_dados(arquivo_entrada, arquivo_saida_completa, arquivo_amostra, tamanho_amostra=0.1):
    """
    Carrega um arquivo JSON, limpa linhas com campos vazios (""),
    salva o arquivo limpo completo e gera um subconjunto para treinamento.

    :param arquivo_entrada: Nome do arquivo JSON de entrada.
    :param arquivo_saida_completa: Nome do arquivo JSON de saída limpo completo.
    :param arquivo_amostra: Nome do arquivo JSON da amostra para treino.
    :param tamanho_amostra: Fração (float) ou número (int) de linhas para a amostra.
    """
    
    if not os.path.exists(arquivo_entrada):
        print(f"ERRO: O arquivo de entrada '{arquivo_entrada}' não foi encontrado.")
        print("Crie o arquivo ou ajuste o nome da variável ARQUIVO_ENTRADA.")
        return

    # 1. Leitura do Arquivo JSON
    print(f"Carregando dados de: {arquivo_entrada}...")
    try:
        df = pd.read_json(arquivo_entrada, lines=True)
        print(f"Total de linhas carregadas: {len(df)}")
    except Exception as e:
        print(f"ERRO ao ler o arquivo JSON: {e}")
        return

    # 2. Pré-processamento: Converter Strings Vazias ("") para Nulos (NaN)
    # Isso garante que o .dropna() identifique as strings vazias como nulas.
    # Usamos regex para cobrir strings vazias e strings com apenas espaços.
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # 3. Análise e Contagem de Nulos
    linhas_nulas = df.isnull().any(axis=1).sum()
    
    print("\n--- Análise de Nulos ---")
    print(f"Linhas com pelo menos um campo vazio/nulo: {linhas_nulas}")
    
    # 4. Remoção de Linhas Nulas
    df_limpo = df.dropna(how='any')
    
    print(f"Linhas removidas: {len(df) - len(df_limpo)}")
    print(f"Linhas restantes (Dados Limpos): {len(df_limpo)}")

    # 5. Salvando o Arquivo Limpo Completo
    # 'orient="records"' mantém o formato de lista de dicionários original.
    df_limpo.to_json(arquivo_saida_completa, orient="records", indent=Nome, lines=True)
    print(f"\n✅ Dados limpos salvos em: '{arquivo_saida_completa}'")

    # --- Parte 6: Geração da Amostra para Treinamento ---
    
    # Gerar a amostra para treino usando o método .sample()
    if tamanho_amostra < 1: # Se for uma fração (ex: 0.1 para 10%)
        amostra_df = df_limpo.sample(frac=tamanho_amostra, random_state=42)
        amostra_tipo = f"{tamanho_amostra*100:.0f}%"
    else: # Se for um número exato de linhas
        amostra_df = df_limpo.sample(n=int(tamanho_amostra), random_state=42)
        amostra_tipo = f"{int(tamanho_amostra)} linhas"
        
    # Salvando a Amostra
    amostra_df.to_json(arquivo_amostra, orient="records", indent=4)
    
    print("\n--- Amostragem para Treinamento ---")
    print(f"Tamanho da Amostra gerada ({amostra_tipo}): {len(amostra_df)}")
    print(f"✅ Amostra para treinamento salva em: '{arquivo_amostra}'")

# --- Execução da Função ---

# Defina aqui o tamanho da sua amostra:
# Ex: 0.1 para 10% do total
# Ex: 1000 para 1000 linhas
TAMANHO_DA_AMOSTRA = 0.1

limpar_e_amostrar_dados(ARQUIVO_ENTRADA, ARQUIVO_SAIDA_COMPLETA, ARQUIVO_AMOSTRA_TREINO, TAMANHO_DA_AMOSTRA)
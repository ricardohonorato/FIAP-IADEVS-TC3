import os
import json
import random
import pandas as pd
import numpy as np

# --- Definição dos Arquivos ---
ARQUIVO_ENTRADA = "/workspaces/python/dataset_alpaca_50k/trn.json"
ARQUIVO_SAIDA_COMPLETA = "dados_limpos_stream.jsonl"
ARQUIVO_AMOSTRA_TREINO = "amostra_treino_stream_filtrada.jsonl" # Renomeado para indicar a filtragem

# Defina a taxa de amostragem (Ex: 0.1 para 10% do total)
TAMANHO_DA_AMOSTRA = 0.044 # 1% do total 

# CAMPOS ESSENCIAIS PARA O TREINAMENTO
CAMPOS_TREINO = ["title", "content"] 
# NOTA: O campo 'input' é comumente usado em datasets como o Alpaca.
# Se precisar, adicione-o aqui: CAMPOS_TREINO = ["instruction", "input", "output"]

def limpar_e_amostrar_stream(arquivo_entrada, arquivo_saida_completa, arquivo_amostra, taxa_amostragem):
    
    if not os.path.exists(arquivo_entrada):
        print(f"ERRO: O arquivo de entrada '{arquivo_entrada}' não foi encontrado.")
        return

    # -----------------------------------------------
    # 1. PROCESSAMENTO COMPLETO (LIMPEZA E SALVAMENTO)
    # -----------------------------------------------
    
    total_lido = 0
    total_limpo = 0
    amostras_para_treino = []
    
    print(f"Iniciando processamento em modo streaming (linha por linha) de: {arquivo_entrada}...")
    
    # Abrindo o arquivo de entrada para leitura e o arquivo de saída para escrita
    with open(arquivo_entrada, 'r', encoding='utf-8') as f_in, \
         open(arquivo_saida_completa, 'w', encoding='utf-8') as f_out:

        for linha in f_in:
            total_lido += 1
            
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue 
            
            # Verifica se algum valor em qualquer campo é nulo ou string vazia
            campos_invalidos_completo = False
            for valor in registro.values():
                if valor is None or (isinstance(valor, str) and not valor.strip()):
                    campos_invalidos_completo = True
                    break
            
            if not campos_invalidos_completo:
                total_limpo += 1
                
                # Escreve o objeto JSON completo (limpo) no arquivo de saída
                f_out.write(json.dumps(registro) + '\n')
                
                # -----------------------------------------------
                # 2. GERAÇÃO DA AMOSTRA (Amostragem estatística e FILTRAGEM DE CAMPOS)
                # -----------------------------------------------
                
                # Decide se este registro limpo fará parte da amostra de treino
                if random.random() < taxa_amostragem:
                    
                    # PASSO CHAVE: Cria um novo dicionário SÓ com os campos desejados
                    registro_filtrado = {}
                    campos_ausentes = False
                    
                    for campo in CAMPOS_TREINO:
                        if campo in registro and registro[campo].strip():
                            registro_filtrado[campo] = registro[campo]
                        else:
                            # Caso um dos campos essenciais tenha sido ignorado na primeira limpeza 
                            # (ou não exista), descartamos a linha para o treino
                            campos_ausentes = True
                            break

                    # Adiciona à lista de amostras APENAS se todos os campos desejados estiverem presentes e não vazios
                    if not campos_ausentes:
                        amostras_para_treino.append(registro_filtrado)
            
            # A memória é liberada após cada iteração do loop
            
    # -----------------------------------------------
    # 3. SALVAR A AMOSTRA FINAL PARA TREINO
    # -----------------------------------------------
    
    # Para garantir o formato JSONL para a amostra de treino
    with open(arquivo_amostra, 'w', encoding='utf-8') as f_sample:
        for item in amostras_para_treino:
            f_sample.write(json.dumps(item) + '\n')


    print("\n--- Resultados do Processamento ---")
    print(f"Total de linhas lidas: {total_lido}")
    print(f"Total de linhas salvas (limpas - completo): {total_limpo}")
    print(f"Linhas descartadas (vazias/nulas): {total_lido - total_limpo}")
    print(f"✅ Dados limpos completos salvos em: '{arquivo_saida_completa}'")
    print(f"Tamanho da Amostra gerada: {len(amostras_para_treino)}")
    print(f"✅ Amostra para treinamento salva em: '{arquivo_amostra}' (Contém apenas os campos: {CAMPOS_TREINO})")

# --- Execução da Função ---
limpar_e_amostrar_stream(ARQUIVO_ENTRADA, ARQUIVO_SAIDA_COMPLETA, ARQUIVO_AMOSTRA_TREINO, TAMANHO_DA_AMOSTRA)
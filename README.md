# FIAP-IADEVS-TC3
TECH CHALLENGE 3


Equipe: <br> 
João Helton RM364239 <br>
João Almeida Furtado Neto RM364164 <br>
Murilo Polli RM364642 <br>
Rafael Pinheiro RM363960 <br> 
Ricardo Honorato RM364026 <br>


VIDEO Youtube Explicativo:  LINK <br>
  
Arquivos:<br>
Na pasta TC3-CleanerDataSet - O script em python para tratamento dos dados e limpeza dos mesmos.
dataset_alpaca_50k.zip - Dataset utilizado para treinar o modelo.


Problemática 1:<br><br>
Limpeza e tratamento de dados: Usando DEVCONTAINER na maquina local <br>
Na primeira abordagem: <br>
Chegamos ao limite da memória : <br>
<img width="733" height="379" alt="image" src="https://github.com/user-attachments/assets/0550a960-45ce-4b36-9827-b2def6a74343" /> <br> <br> <br>
Erro 1:
<img width="752" height="73" alt="image" src="https://github.com/user-attachments/assets/326930a9-2dcf-4a58-ad6f-8a258bd6f4c2" /> <br><br><br>
Erro 2:
<img width="739" height="64" alt="image" src="https://github.com/user-attachments/assets/7075bb15-4824-4955-a93e-54af0c7a373e" /> <br><br><br>

Na segunda abordagem mudamos a estratégia e processamos linha a linha. <br>
<img width="745" height="396" alt="image" src="https://github.com/user-attachments/assets/78a2dc84-9072-4c46-b70f-e4d93a15b5a0" />

Com isso conseguimos processar o arquivo: (abaixo segue um exemplo do processamento) <br>
<img width="752" height="114" alt="image" src="https://github.com/user-attachments/assets/ca87ba70-ccc0-4589-9a28-01b1b008c257" /> <br>


Problemática 2:<br><br>
Treinamento do modelo: <br>
Usamos o unsloth, que é um framework Python de código aberto projetado para tornar o ajuste fino (fine-tuning) e o treinamento de grandes modelos de linguagem (LLMs) significativamente mais rápidos e com uso mais eficiente da memória. <br>

A seguir iremos falar sobre as variaves, importante destacar que para redução de custos limitamos o número de procesadores para 2. <br><br>

per_device_train_batch_size	Inteiro	O tamanho do lote de treinamento por dispositivo (GPU). Neste caso, 2. <br>
gradient_accumulation_steps	Inteiro	O número de etapas de retropropagação (backward passes) antes de realizar uma etapa de otimização (update). Ajuda a simular um lote maior. Aqui é 4. <br>
warmup_steps	Inteiro	O número de etapas para aumentar a taxa de aprendizado (learning rate) linearmente a partir de 0. Aqui é 5. <br>
max_steps	Inteiro	O número máximo de etapas de otimização (não épocas) para executar o treinamento. Neste caso, 60. <br>
learning_rate	Float	A taxa de aprendizado inicial a ser usada para o otimizador. Aqui é 2e-4 ((2×10  
−4
 )). <br>
fp16	Booleano	Se deve ser usada precisão de ponto flutuante de 16 bits (float16) para treinamento. Definido como True se o bfloat16 não for suportado. <br>
bf16	Booleano	Se deve ser usada precisão de ponto flutuante de 16 bits Brain Float (bfloat16) para treinamento. Definido como True se o bfloat16 for suportado. <br>
logging_steps	Inteiro	A frequência com que o log de informações (como a perda de treinamento) deve ser relatado. Aqui é 1 (a cada etapa). <br>
optim	String	O otimizador a ser usado. Neste caso, é o adamw_8bit, uma versão otimizada com 8 bits que economiza memória. <br>
weight_decay	Float	O fator de penalização de L2 aplicado aos pesos do modelo para evitar overfitting. Aqui é 0.01. <br>
lr_scheduler_type	String	O tipo de agendador (scheduler) para ajustar a taxa de aprendizado durante o treinamento. Aqui é linear. <br>
seed	Inteiro	A semente (seed) aleatória para garantir a reprodutibilidade dos resultados. Neste caso, é 3407. <br>
output_dir	String	O diretório onde os pontos de verificação (checkpoints) e os resultados do treinamento serão salvos. Aqui é outputs. <br>
report_to	String/Lista	Para onde o treinamento deve ser reportado (por exemplo, "wandb", "tensorboard"). Neste caso, é none (nenhum). <br>









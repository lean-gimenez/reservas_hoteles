# ANÁLISE DE DADOS: CANCELAMENTOS DE RESERVAS HOTELARIAS

## Descrição
Análise exploratória do dataset 'hotel_bookings.csv', que contém informações sobre reservas de hotéis. O objetivo é identificar padrões de comportamento nos cancelamentos por meio do treinamento de um modelo capaz de prever e classificar o status das reservas. O processo inclui limpeza do dataframe, estatísticas e visualizações.


## Estrutura
Desafio_extra/
├──hotel_bookings.csv   # Dataset com reservas
├──analise.py           # Script com a análise completa do dataset
├──requirements.txt     # Lista de dependências
├──Graficos/            # Pasta com graficos gerados
│       ├──boxplots.png
│       ├──cancelamentos_alvo.png
│       ├──importancia_variaveis.png
│       ├──matriz_de_confucao.png
│       └──outliers.png
└──README.md            # Esta documentação


## Instruções de execução
### Executar o código Python localmente
1. Descompacte o arquivo .rar no seu computador
2. Abra o terminal na pasta do projeto
3. Instale as dependências listadas no projeto:
    pip install -r requirements.txt
4. Execute o script principal:
    python analise.py


## Funcionalidades principais
1. Carregamento de dados: desde o arquivo CSV
2. Limpeza e Tratamento: deteção e eliminação de duplicados e nulos
3. Feature Engineering: Criação de variáveis derivadas estratégicas (has_children, has_cancelled, room_changed).
4. Modelagem Preditiva: Separação treino/teste (80/20), pré-processamento com One-Hot Encoding e treinamento da Árvore de Decisão com class_weight='balanced'.
5. Avaliação de Desempenho: Cálculo de Acurácia, Matriz de Confusão, Precisão, Recall e F1-Score, além da análise de Feature Importance.
6. Visualização: gráficos de barras, pizza e mapas de calor


## Dependências
* pandas 1.5.0+: Manipulação e análise de dados
* numpy (1.21.0+): Operações vetoriais e condicionais nativas.
* matplotlib 3.5.0+: Gráficos e visualizações básicas
* seaborn 0.12.0+: Visualizações estatísticas avançadas
* scikit-learn (1.1.0+): Pré-processamento, divisão dos dados e algoritmo de Árvore de Decisão.


## Metodologia de Análise
Exploração inicial: para verificar a estrutura do dataframe e os tipos de dados (numericos/categoricos).
Limpeza: Tratamento de nulos, eliminação de duplicatas e remoção de colunas
Tratamento de Outliers: Filtragem de valores inválidos ou extremos na variável adr (0 < adr < 1000).
Análise de Distribuição e Relações: Avaliação do impacto das variáveis categóricas e numéricas na taxa de cancelamento.
Modelagem e Avaliação: Treinamento do modelo supervisionado e interpretação das métricas.


## Estatísticas do Dataset

### 1. Visão Geral Comparativa
| Métrica    | Dataset Original | Dataset Final (Tratado) |
| ---------- | ---------------- | ----------------------- |
| Registros  | 119390           | 85616                   |
| Colunas    | 32               | 43 (apos OHE)           |
| Duplicados | 31994            | 0 (Removidos)           |
| Nulos      | 94786            | 0 (Tratados/Imputados)  |

### 2. Distribuição da Variável-Alvo (`is_canceled`)
Total = 85616
A variável-alvo apresenta um desbalanceamento natural entre reservas confirmadas e canceladas:
* Não Cancelado (0): 61777 reservas (72.2%)
* Cancelado (1): 23839 reservas (27.8%)
* Proporção Conjunto de Teste (y_test): 12356 (0) vs. 4768 (1) [17124 registros totais]

NOTA:
    A distribuição inicial da variável-alvo no dataset sem duplicados (87396 registros) era de 72.5% não cancelados (63371 registros) e 27.5% cancelados (24025 registros). Após o filtro de consistência na tarifa diária (0 < adr < 1000), a base final de modelagem fechou em 85616 registros (72.2% não cancelados vs 27.8% cancelados).

### 3. Mudanças nos dados
Data Leakage
    Como a variável-alvo é 'is_canceled', as colunas 'reservation_status' e 'reservation_status_date' são um problema de Data Leakage para o modelo, pois contêm a resposta futura (Check-Out, Canceled, No-Show e a data de status).
    Por isso, ambas são excluídas do DataFrame.

É necessario descartar outliers das colunas
    'adr' (Average Daily Rate):
        Presença de valor negativo (min = -6.38), o que pode ser um erro de sistema.
        Presença de um valor máximo extremo (max = 5400.00), desconectado do 3º quartil (134.00).
        Decisão: Filtrar o DataFrame mantendo apenas valores válidos de ADR (0 <= adr < 1000).

    'stays_in_week_nights', 'stays_in_weekend_nights', 'lead_time':
        Apresentam valores máximos elevados, mas pode ser por estadias longas e reservas antecipadas reais do setor hoteleiro (não serão considerados erros).
    
    'required_car_parking_spaces', 'previous_bookings_not_canceled', 'total_of_special_requests':
        Apresentam distribuição concentrada no zero (zero-inflated). Embora a média e a mediana difiram, esses valores raros podem ser altamente preditivos de não-cancelamento.
        Decisão: Manter as colunas.

### 4. Modelagem preditiva e Feature Engineering (Variáveis Derivadas):
  1. has_children: Indicador binário derivado da soma de 'children' e 'babies'.
  2. has_cancelled: Indicador binário derivado de 'previous_cancellations'.
  3. room_changed: Indicador binário comparando 'reserved_room_type' e 'assigned_room_type'.


## Insights Gerados

1. Crianças
    A presença de crianças tem influência no cancelamento. Quem viaja com crianças cancela 7.55% mais.

2. Alteração do quarto (Check-in)
    Qualquer alteração no quarto designado (room_change_type != 'Sem Troca') reduz a taxa de cancelamento de 31.51% para menos de 5%.
    Não há uma diferença significativa na taxa de cancelamento entre receber um Upgrade ou um Downgrade.
    Isso possivelmente é porque a reatribuição do quarto ocorre quando o cliente está fazendo o check-in, então pode ser menos provável que o cliente deseje cancelar.

3. Antecedência da reserva (lead_time)
    Quanto maior a antecedência da reserva, maior é a probabilidade de cancelamento.

4. Histórico Reincidente
    Clientes que já cancelaram pelo menos uma reserva no passado apresentam uma taxa de cancelamento de 68%, em comparação com 27% daqueles sem histórico.

5. Tipo de Depósito
    As reservas que não precisaram depósito e as que têm reembolso têm menos de 27% de cancelamentos. Enquanto que as reservas que não têm reembolso têm um 94,7% de cancelamentos.

6. Segmentos de Mercado
    Os segmentos que mais cancelaram suas reservas foram Agência Online de Viagens (35,35%) e Grupos (27,01%).


## Conclusões Finais
O modelo de Árvore de Decisão revelou que três fatores concentram mais de 50% do poder preditivo das cancelações: o tempo de antecedência da reserva (lead_time), a solicitação de vaga de garagem (required_car_parking_spaces) e o número de pedidos especiais (total_of_special_requests).

A presença da variável derivada has_cancelled no Top 6 confirma a relevância do histórico do cliente. Para a gestão hoteleira, esses achados sugerem focar políticas de retenção e confirmação em reservas feitas com grande antecedência via agências online (Online TA) e sem solicitações especiais associadas.


## Autor
Leandro Julián Giménez


## Contato
gimenez.leandro.j@gmail.com

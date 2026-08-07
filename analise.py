import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# =============================================
# CONFIGURAÇÕES GERAIS
# =============================================

#print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
pd.set_option('display.float_format', '{:,.2f}'.format)

print("ANÁLISE DE RESERVAS DE HOTEIS")

# =============================================
# 1. CARREGAMENTO E LIMPEZA DE DADOS
# =============================================
print("")
print("=" * 80)
print("1. CARREGAMENTO E LIMPEZA DE DADOS")
print("=" * 80)

# CARREGAR DATASET E CONVERTIR EM DATAFRAME
def carregar_dados():
    dataset = 'hotel_bookings.csv'
    df_original = pd.read_csv(dataset) #encoding='latin-1'
    print(f"Dataset '{dataset}' carregado exitosamente.")
    return df_original

# ASIGNAR DATAFRAME A VARIÁVEL 'df'
df_original = carregar_dados()

# INFO BASICA DO DF

print("")
print(f"O Data Frame (df) contem {df_original.shape[0]} filas e {df_original.shape[1]} colunas.")
print("")

# DUPLICADOS 

print(f"Duplicados:\n{df_original.duplicated().sum()}\nExcluídos do df.")
# NOTA:
# O DF tem 31994 linhas duplicadas

df = df_original.drop_duplicates() # Linhas duplicadas excluídas

# NULOS
print("")
nulos = df.isnull().sum()
print(f"Nulos:\n{nulos[nulos > 0]}")

# - A coluna 'children' tem 4 dados nulos.
# -- Problema: é incomum que os dados de "children" sejam do tipo float.
# -- Solução: imputar nulos com a Mediana e passar tudo a int.

# - A coluna 'country' tem 452 nulos.
# -- Solução: imputar nulos com 'Unknown'.

# - A coluna 'agent' tem 12193 nulos. 
# -- Problema: todos os dados são float com decimais "0" (zero).
# -- Solução: imputar nulos com 0 e passar tudo a str porque é o 'nome' do agente.

# - A coluna 'company' tem 82137 nulos. 
# -- Problema: todos os dados são float com decimais "0" (zero).
# -- Solução: imputar nulos com 0 e passar tudo a str porque é o 'nome' da compania.

# COLUNA 'children'
df['children'] = df['children'].fillna(df['children'].median()).astype(int)

# COLUNA 'country'
df['country'] = df['country'].fillna('Unknown')

# COLUNA 'agent'
df['agent'] = df['agent'].fillna(0).astype(int).astype(str)

# COLUNA 'company'
df['company'] = df['company'].fillna(0).astype(int).astype(str)

# ELIMINAÇÃO DE COLUNAS COM DATA LEAKAGE

# Como a variável-alvo é 'is_canceled', as colunas 'reservation_status'
# e 'reservation_status_date' são um problema de Data Leakage para o modelo,
# pois contêm a resposta futura (Check-Out, Canceled, No-Show e a data de status).
# Por isso, ambas são excluídas do DataFrame.

df = df.drop(columns=['reservation_status', 'reservation_status_date'])

print("")
print(f"DataFrame atualizado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
print("")
print("Resumo estatístico das variáveis numéricas:")
print(df.describe().T[['min', 'mean', 'max']])

# COLUNAS NUMERICAS E CATEGORICAS

colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
colunas_categoricas = df.select_dtypes(include=['object', 'str']).columns.tolist()

# Excluímos a variável-alvo 'is_canceled' da lista de numéricas
colunas_numericas.remove('is_canceled')

print("")
print(f"As colunas numéricas são {len(colunas_numericas)}\n{colunas_numericas}")
print("")
print(f"As colunas categóricas são {len(colunas_categoricas)}\n{colunas_categoricas}")
print("")
print("A variável-alvo é 'is_canceled' e é numerica.")

# =============================================
# 2. AED
# =============================================
print("")
print("=" * 80)
print("2. Análise Exploratória de Dados")
print("=" * 80)

# VARIÁVEL-ALVO is_canceled: DISTRIBUIÇÃO
#print("")
#print("Distribuição dos Cancelamentos (variável-alvo)")
#print(df['is_canceled'].value_counts())


# HIPÓTESE 1:
# A quantidade de dias na sala de espera tem influença sobre a cancelação
espera = df.groupby('days_in_waiting_list')['is_canceled'].agg(
    total_reservas='count',
    total_cancelados='sum',
    taxa_cancelamento='mean'
    )
espera['taxa_cancelamento'] *= 100
print("")
print("Dias na sala de espera X Cancelamento:")
print(espera[espera['total_reservas']>30].sort_values(by='taxa_cancelamento', ascending=False))

# A maior quantidade de cancelamentos são aos 0 dias de ficar na lista de espera.
# Então podemos dizer que não tem influença no cancelamento.

#-------------------------------------------------------------
# HIPÓTESE 2:
# A presença de crianças pode afetar os cancelamentos.
df['children_total'] = df['children'] + df['babies']
df['has_children'] = np.where(df['children_total'] > 0, 1, 0)

criancas = df.groupby('has_children')['is_canceled'].agg(
    total_reservas = 'count',
    total_cancelados = 'sum',
    taxa_cancelamento = 'mean'
)
criancas['taxa_cancelamento'] *= 100
print("")
print("Crianças X Cancelamento:")
print(criancas)

# Das 9103 famílias que têm crianças, 3118 cancelaram (34.25%),
# por enquanto das 78293 reservas sem crianças, cancelaram 20907 (26.70%)
# Então podemos dizer que as famílias com crianças TÊM influença no cancelamento.

#-------------------------------------------------------------
# HIPÓTESE 3:
# O troco de um quarto por outro pode afetar os cancelamentos.
df['room_changed'] = np.where(df['reserved_room_type'] == df['assigned_room_type'], 0, 1)
quarto = df.groupby('room_changed')['is_canceled'].agg(
    total_reservas = 'count',
    total_cancelados = 'sum',
    taxa_cancelamento = 'mean'
)
quarto['taxa_cancelamento'] *= 100
print("")
print("Troca de quarto X Cancelamento:")
print(quarto)

# Apenas 4.71% cancelou a reserva quando o quarto era redesignado.
# Mas 31.51% cancelou sem a troca do quarto.
# É possivel que a troca de quarto ajude a prevenir os cancelamentos.


# Ver os valores das colunas 'reserved_room_type' e 'assigned_room_type'
print(f"Quartos reservados:\n{df['reserved_room_type'].value_counts()}")
print(f"Quartos designados\n{df['assigned_room_type'].value_counts()}")

# Os quartos têm uma classificação de A, B, C, D, E, F, G, H, I, K, L, P
# Verificar como funcionan as jerarquias dos quartos comparando tipo de quarto e ADR (valor promedio).

# Calcular o preço de cada quarto
mapa_quartos_reservados = df.groupby('reserved_room_type')['adr'].mean()
mapa_quartos_designados = df.groupby('assigned_room_type')['adr'].mean()

# Mapear os preços em colunas novas
df['valor_reservado'] = df['reserved_room_type'].map(mapa_quartos_reservados)
df['valor_designado'] = df['assigned_room_type'].map(mapa_quartos_designados)

# Condições
condicoes = [
    df['reserved_room_type'] == df['assigned_room_type'], # 1. Sem troca de quarto
    df['valor_designado'] > df['valor_reservado'],        # 2. Upgrade
    df['valor_designado'] < df['valor_reservado']         # 3. Downgrade
]
opcoes = [
    'Sem Troca de Quarto',
    'Upgrade',
    'Downgrade'
]

# Comparar os preços promedio dos quartos
df['room_change_type'] = np.select(condicoes, opcoes, default = 'Mesmo Nível')

troca_de_quartos = df.groupby('room_change_type')['is_canceled'].agg(
    total_reservas = 'count',
    total_cancelados = 'sum',
    taxa_cancelamento = 'mean'
)
troca_de_quartos['taxa_cancelamento'] *= 100
print("")
print("Análise de Troca de Quartos X Cancelamento:")
print(troca_de_quartos)

# Os cancelamentos por Upgrade (4,88%) e por Downgrade (3.73%) são menores do que "Sem Troca de Quarto"
# Qualquer alteração no quarto designado (room_change_type != 'Sem Troca') 
# reduz a taxa de cancelamento de 31.51% para menos de 5%.
# Não há uma diferença significativa na taxa de cancelamento entre receber um Upgrade ou um Downgrade.
# Isso possívelmente é porque a reatribução do quarto ocorre quando o cliente está fazendo o check-in,
# então pode ser menos provável que o cliente deseje cancelar.

#-------------------------------------------------------------
# HIPÓTESE 4:
# O tempo de antecedência das reservas pode afetar os cancelamentos.

# Agrupar cancelamentos por antecedência
antecedencia = df.groupby('is_canceled')['lead_time'].describe()

print("")
print("Antecedência das reservas X Cancelamento")
print(antecedencia)

# 25% das reservas não canceladas foram feitas com 7 ou menos dias de antecedência.
# 25% das reservas canceladas foram feitas com 32 ou menos dias de antecedência. 
# Reservas feitas com meses de antecedência têm um risco muito mais elevado 
# de alteração de planos do que reservas de última hora (feitas na mesma semana do check-in).

#-------------------------------------------------------------
# HIPÓTESE 5:
# As pessoas com maior numero de cancelamentos previos são mais prováveis de cancelar novamente.

# Taxa de cancelamentos previos X Cancelamentos
df['has_cancelled'] = np.where(df['previous_cancellations'] > 0, 1, 0)
cancelamentos_previos = df.groupby('has_cancelled')['is_canceled'].agg(['count', 'mean'])
print("")
print("Taxa de cancelamentos X Cancelamentos")
print(cancelamentos_previos)

# O histórico de cancelamentos anteriores (has_cancelled) é um forte indicador de reincidência.
# Clientes que já cancelaram pelo menos uma reserva no passado apresentam uma taxa de cancelamento de 68%, 
# em comparação com 27% daqueles sem histórico.

#-------------------------------------------------------------
# HIPÓTESE 6:
# O tipo de deposito pode afetar os cancelamentos.

deposito = df.groupby('deposit_type')['is_canceled'].agg(
    total_reservas = 'count',
    total_cancelados = 'sum',
    taxa_cancelamento = 'mean'
)
deposito['taxa_cancelamento'] *= 100
print("")
print("Tipo de deposito X Cancelamento")
print(deposito)

# As reservas que não precisaram deposito e as que têm reembolso têm menos de 27% de cancelamentos.
# Enquanto que as reservas que não têm reembolso têm um 94,7% de cancelamentos.
# Talvez as agencias de viagens compram pacotes de viagens com muita antecedência
# por um preço mais barato, e depois possívelmente não têm clientes para esses pacotes,
# mas não representa um gasto grande para elas.

#-------------------------------------------------------------
# HIPÓTESE 7:
# Pode ter algum segmento que acostume cancelar.

segmento = df.groupby('market_segment')['is_canceled'].agg(
    total_reservas = 'count',
    total_cancelados = 'sum',
    taxa_cancelamento = 'mean'   
)
segmento['taxa_cancelamento'] *= 100
print("")
print("Segmento X Cancelamentos")
print(segmento.sort_values(by='taxa_cancelamento', ascending=False))

# Os segmentos que mais cancelaram suas reservas foram Agência Online de Viagens (35,35%) e Grupos (27,01%).

deposito_segmento = df.groupby(['market_segment', 'deposit_type'])['is_canceled'].agg(
    total_reservas = 'count',
    total_cancelados = 'sum',
    taxa_cancelamento = 'mean'   
).reset_index()
deposito_segmento['taxa_cancelamento'] *= 100

print("")
print("Segmento X Tipo de depósito X Cancelamentos")
filtro_deposito_taxa_cancelamento = (deposito_segmento['deposit_type'] == 'Non Refund') & (deposito_segmento['taxa_cancelamento'] > 25) & (deposito_segmento['total_reservas'] > 30)
print(deposito_segmento[filtro_deposito_taxa_cancelamento].sort_values(by='taxa_cancelamento', ascending=False))

# Os cancelamentos massivos em tarifas Non Refund ocorrem exclusivamente em canais intermediários ou institucionais: 
# Offline TA/TO (98.95%), Groups (95.90%) e Corporate (70.31%).

# CONCLUSÃO

# Desestimar:
# - Dias na lista de espera

# Ficar com:
# - Has cancelled
# - Antecedência da reserva
# - Tipo de deposito
# - Segmento
# - Troca de quarto
# - Crianças


# =============================================
# 3. PRE-PROCESSAMENTO
# =============================================
print("")
print("=" * 80)
print("3. PRE-PROCESSAMENTO")
print("=" * 80)

# INFO DAS COLUNAS
print(df.info())

# DIAGNÓSTICO DE OUTLIERS
colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
colunas_binarias_excluidas = ['is_canceled', 'is_repeated_guest', 'has_children', 'has_cancelled', 'room_changed']
colunas_outliers = [col for col in colunas_numericas if col not in colunas_binarias_excluidas]

# Graficar com boxplot para ver outliers
plt.figure(figsize=(15,10))
for i, col in enumerate(colunas_outliers, 1):
    plt.subplot(5, 5, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col, fontsize=10)
    plt.ylabel('')
plt.tight_layout()
plt.show()

# Colunas com outliers
verificar_col = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
                 'required_car_parking_spaces', 'previous_bookings_not_canceled', 'total_of_special_requests', 'adr']
for c in verificar_col:
    print("")
    print(c, df[c].describe())

# É necessario descartar outliers das coluna 'adr' (Average Daily Rate)
# Não precisam mudanças:
# 'stays_in_week_nights', 'stays_in_weekend_nights', 'lead_time',
# 'required_car_parking_spaces', 'previous_bookings_not_canceled', 'total_of_special_requests':


# Filtro de coluna 'adr'
df = df[(df['adr'] > 0) & (df['adr'] < 1000)].copy()



# DISTRIBUIÇÃO FINAL DA VARIÁVEL-ALVO is_canceled
print("")
print("Distribuição Final dos Cancelamentos (Dataset Tratado com 85616 registros):")
print(df['is_canceled'].value_counts())
plt.figure(figsize=(8,5))
plt.pie(df['is_canceled'].value_counts(), 
        labels=['Não Cancelado', 'Cancelado'], 
        autopct='%1.1f%%', 
        startangle=90)
plt.title("Distribuição Final dos Cancelamentos (variável-alvo)")
plt.show()


# ELIMINAR COLUNAS NÃO RELEVANTES OU REDUNDANTES

# A coluna 'days_in_waiting_list' porque não mostrou nenhuma influência.
# As colunas 'adults', 'children', 'babies', 'children_total' estão resumidas na coluna 'has_children'.
# As colunas 'valor_reservado', 'valor_designado', 'reserved_room_type', 'assigned_room_type', 
# - resumidas em 'room_changed' e 'room_change_type'.
# - Mas essas colunas não podem ser utilizadas para prever, porque a reatribução de quartos acontece depois da reserva,
# - então a info da coluna 'assigned_room_type' não é certero (pode trocar ainda).
# As colunas 'agent', 'company', 'country' por ser muito cardinais e a codificação delas
# - geraria um aumento excessivo de colunas esparsas, o que sujaria os dados.
# As colunas 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
# - porque o mês ('arrival_date_month') tem mais info sobre a sazonalidade.
# A coluna 'previous_cancellations' porque está resumida em 'has_cancelled'.

colunas_descartadas = ['days_in_waiting_list',
                       'adults', 'children', 'babies', 'children_total', 
                       'valor_reservado', 'valor_designado', 'reserved_room_type', 'assigned_room_type', 'room_changed', 'room_change_type',
                       'agent', 'company', 'country',
                       'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
                       'previous_cancellations'
                        ]

# SPLIT X e y is_canceled
X = df.drop(columns=colunas_descartadas + ['is_canceled']).copy()
y = df['is_canceled'].copy()

print("")
print(f"As {X.shape[1]} colunas selecionadas para a variável de treinamento do modelo:")
print(list(X.columns))


# CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS (ONE-HOT ENCODING)

# Identificar as colunas categóricas em X (vamos reescrever a variável anterior, não é um problema para essa etapa do processamento)
colunas_categoricas = X.select_dtypes(include=['object', 'str']).columns.tolist()

# Instanciar o objeto OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')

# Aplicar transformações
X_categoricas_encoded = ohe.fit_transform(X[colunas_categoricas])

# UNIR TODAS AS COLUNAS EM DF X_encoded

# Identificar as colunas categóricas em X (vamos reescrever a variável anterior, não é um problema para essa etapa do processamento)
colunas_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Unir tudo
X_encoded = pd.concat([X_categoricas_encoded, X[colunas_numericas]], axis=1)

print("")
print("Codificação e união concluídas com sucesso.")
print(f"Dimensão de X_encoded: {X_encoded.shape}")
X_encoded.info()

# DIVISÃO EM TREINO E TESTE (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("")
print("Divisão em Treino e Test concluída com sucesso.")
print(f"Variável de treino (X_train): {X_train.shape[0]} amostras (80%).")
print(f"Variável de test (X_test): {X_test.shape[0]} amostras (20%).")


# ==============================================================================
# 4. TREINAMENTO DO MODELO (DECISION TREE)
# ==============================================================================
print("")
print("=" * 80)
print("4. TREINAMENTO E AVALIAÇÃO DO DECISION TREE")
print("=" * 80)

# Instanciar o modelo
tree = DecisionTreeClassifier(
    max_depth=10, # Para evitar overfitting
    random_state=42,
    class_weight='balanced' # Balance entre não cancelamentos (grupo maior) e cancelamentos (grupo menor)
)

# Treinar o modelo com os dados de treino
tree.fit(X_train, y_train)
print("")
print("Modelo 'Decision Tree' treinado com sucesso.")

# Fazer predições nos dados de test
y_pred = tree.predict(X_test)

# ==============================================================================
# 5. AVALIAÇÃO DE DESEMPENHO E MÉTRICAS
# ==============================================================================
print("")
print("=" * 80)
print("5. MÉTRICAS DE AVALIAÇÃO")
print("=" * 80)

# Relatório de Classificação (Acurácia, Precisão, Recall e F1-Score)
print("")
print("Relatório de Classificação:")
relatorio = classification_report(y_test, y_pred, target_names=['Não Cancelado', 'Cancelado'])
print(relatorio)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,5))
sns.heatmap(
    cm, 
    annot=True, # apresentar valores no grafico
    fmt='d', # decimal integer
    cmap='Blues',
    xticklabels=['Não Cancelado', 'Cancelado'],
    yticklabels=['Não Cancelado', 'Cancelado']
)
plt.title("Matriz de Confução")
plt.xlabel('Predição do Modelo')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.show()

# ANÁLISE DO RELATÓRIO E MATRIZ DE CONFUSÃO
#
# Matriz de Confução:
#
# Verdadeiros Negativos (VN): 8428 (eram Não Canceladas, previu Não Canceladas) --> ACERTOU
# Falsos Positivos (FP): 3928 (eram Não Canceladas, previu Canceladas) --> ERROU
# Falsos Negativos (FN): 1001 (eram Canceladas, previu Não Canceladas) --> ERROU
# Verdadeiros Positivos (VP): 3767 (eram Canceladas, previu Canceladas) --> ACERTOU
#
# Support --> Casos reais
# - Não Cancelado: 12356 (VN + FP)
# - Cancelado: 4768 (FN + VP)
#
# Predições feitas
# - Não Cancelado: 9429 (VN + FN)
# - Cancelado: 7695 (FP + VP)
#
# Precisão --> Das predições feitas, quantas foram respostas corretas
# - Não Cancelado: 0.89 -- Das 9429 predições como Não Cancelado, acertou 8428. [VN / (VN + FN)]
# - Cancelado: 0.49 -- Das 7695 predições como Cancelado, acertou 3767.         [VP / (VP + FP)]
#
# Recall --> Cancelamentos captados
# - Não Cancelado: 0.68 -- Dos 12356 casos reais, acertou 8428. [VN / (VN + FP)]
# - Cancelado: 0.79 -- Dos 4768 casos reais, acertou 3767.      [VP / (VP + FN)]
# 
# F1-score --> Média harmónica entre precisão e recall [2 * (P / R) / (P + R)]
# - Não Cancelado: 0.77
# - Cancelado: 0.60


# ==============================================================================
# 6. IMPORTÂNCIA DAS VARIÁVEIS (FEATURE IMPORTANCE)
# ==============================================================================
print("")
print("=" * 80)
print("6. ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS (FEATURE IMPORTANCE)")
print("=" * 80)

# Extrair as importâncias das variáveis e os nomes das colunas (de X_train)
importancias = tree.feature_importances_
colunas = X_train.columns

# Criar um DataFrame ordenado do maior para o menor
df_importancia = pd.DataFrame({
    'Variável': colunas,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

# Apresentar as 10 variáveis mais importantes
print("Top 10 variáveis mais importantes:")
print(df_importancia.head(10).to_string(index=False))

# Fazer o gráfico de barras horizontais (Top 10)
plt.figure(figsize=(10,5))
sns.barplot(
    data=df_importancia.head(10),
    x='Importancia',
    y='Variável',
    hue='Variável',
    palette='viridis',
    legend=False
)
plt.title("Top 10 variáveis mais influentes")
plt.xlabel("Pontuação de Importância")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()

# As variáveis mais influentes são
# - Antecedência da reserva (lead_time)
# - Espaços de estacionamento solicitados (required_car_parking_spaces)
# - Pedidos especiais (total_of_special_requests)
# - Segmento: Agência de Viagens Online (market_segment_Online TA)
# - Deposito: Sem Reembolso (deposit_type_Non Refund)
# - Clientes que cancelaram no passado (has_cancelled)
# - Preço diario promédio (adr)
# - Mudanças nas reservas (booking_changes)
# - Cliente: (customer_type_Transient-Party)
# - (customer_type_Transient)

# README
# Antecedência da reserva (lead_time)
# Quanto maior o numero de dias entre a reserva e a data de chegada, maior a incertidumbre.
# As reservas feitas com meses de antecedência estçao sujeitas a cancelamento devido a mudanças de planos.

# Espaços de estacionamento solicitados (required_car_parking_spaces)
# Forte sinal de intenção real.
# Interpretação: Um hóspede que solicita estacionamento 
# geralmente viaja de carro (muitas vezes turismo local ou regional) 
# e assume um compromisso logístico claro com a viagem. 
# A probabilidade de cancelamento neste grupo cai drasticamente.

# Pedidos especiais (total_of_special_requests)
# Comprometimento e interesse do cliente.
# Interpretação: Solicitar cama alta, berço ou um andar específico 
# demonstra que o cliente está planejando ativamente a sua estadia.

# Segmento: Agência de Viagens Online (market_segment_Online TA)
# As agências de viagens online (Booking, Decolar, Skyscanner, etc.) 
# facilitam as reserva por impulso e o cancelamento gratuito, aumentando o risco de cancelamento.
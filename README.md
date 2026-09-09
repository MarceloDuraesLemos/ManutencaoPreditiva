#  UPX 2.0 — Plataforma de Manutenção Preditiva

Projeto de **manutenção preditiva e prognóstico** desenvolvido em Python e Streamlit com o dataset **NASA C-MAPSS FD001**.

A aplicação combina modelos de Machine Learning para estimar **Remaining Useful Life (RUL)**, detectar **anomalias**, acompanhar a **tendência de degradação**, calcular um **Health Score** e gerar uma **prioridade operacional de manutenção** para cada equipamento.

> O módulo atual utiliza motores turbofan simulados do NASA C-MAPSS FD001. Os resultados são experimentais e não devem ser aplicados diretamente a máquinas reais sem treinamento e calibração específicos.

---

##  Objetivo

Transformar séries temporais de sensores em informações úteis para manutenção, respondendo perguntas como:

- Qual equipamento apresenta pior condição?
- Qual possui menor vida útil remanescente estimada?
- Existem sinais de comportamento anômalo?
- A degradação está aumentando?
- Qual equipamento deve ser priorizado para manutenção?

Fluxo geral:

```text
Sensores
   ↓
Pré-processamento
   ↓
┌───────────────┬─────────────────────┐
│ LSTM RUL      │ LSTM Autoencoder    │
└───────┬───────┴──────────┬──────────┘
        │                  │
        │             Anomaly Score
        │                  │
        │              Trend Score
        │                  │
        └────────┬─────────┘
                 ↓
            Health Score
                 ↓
               Status
                 ↓
           Priority Score
                 ↓
          P1 / P2 / P3 / P4
                 ↓
              Dashboard
```

---

##  Modelos utilizados

### 1. RUL — Remaining Useful Life

O modelo principal de prognóstico utiliza uma **LSTM** para estimar a quantidade de ciclos restantes do equipamento.

Configuração principal:

- Dataset: NASA C-MAPSS FD001
- Modelo: LSTM
- Janela temporal: 30 ciclos
- Sensores utilizados: 14
- Target de treinamento: RUL capped em 125 ciclos
- Saída: RUL estimado em ciclos

Resultado integrado:

- **MAE:** aproximadamente `11.123 ciclos`
- **RMSE:** aproximadamente `15.079 ciclos`

> O conjunto oficial de teste foi utilizado durante o desenvolvimento para comparação entre configurações. Portanto, ele não deve ser descrito como um holdout completamente intocado.

### 2. Detecção de anomalias

A detecção de anomalia utiliza um **LSTM Autoencoder**. O modelo aprende uma região proxy de comportamento saudável e mede o erro de reconstrução das janelas futuras.

Durante o desenvolvimento, a região saudável foi definida por:

```text
RUL linear > 125 ciclos
```

Na inferência, o Autoencoder recebe **somente dados dos sensores**.

O erro de reconstrução é convertido em:

```text
Anomaly Score: 0 → 100
```

Quanto maior o score, maior o desvio em relação ao comportamento saudável aprendido.

### 3. Trend Score

A tendência é calculada a partir do erro de reconstrução recente do Autoencoder.

São usadas **10 janelas consecutivas** e uma regressão linear sobre:

```text
erro de reconstrução × ciclo
```

O slope é convertido em:

```text
Trend Score: 0 → 100
```

Valores maiores representam uma tendência de piora mais intensa.

---

##  Health Score

O Health Score representa a condição técnica combinada do equipamento.

```text
0   = pior condição
100 = melhor condição
```

Fórmula:

```text
Health =
0.60 × RUL Score
+ 0.25 × (100 - Anomaly Score)
+ 0.15 × (100 - Trend Score)
```

| Health Score | Status |
|---|---|
| 80–100 | 🟢 SAUDÁVEL |
| 60–79.99 | 🟡 ATENÇÃO |
| 30–59.99 | 🟠 RISCO |
| 0–29.99 | 🔴 CRÍTICO |

---

##  Priority Score

O Priority Score representa **urgência operacional de manutenção**.

```text
Health Risk = 100 - Health Score
RUL Risk    = 100 - RUL Score
```

```text
Priority Score =
0.30 × Health Risk
+ 0.70 × RUL Risk
```

| Priority Score | Classe |
|---|---|
| < 20 | 🟢 P4 — BAIXA |
| 20 a < 45 | 🟡 P3 — MÉDIA |
| 45 a < 75 | 🟠 P2 — ALTA |
| ≥ 75 | 🔴 P1 — IMEDIATA |

---

##  Dashboard Streamlit

A interface possui três áreas principais:

###  Dashboard Geral

Exibe:

- quantidade total de motores;
- análises completas e parciais;
- equipamentos críticos;
- prioridades P1;
- Health Score médio;
- distribuição por condição;
- distribuição por prioridade;
- ranking dos equipamentos mais urgentes;
- identificação separada de equipamentos com análise parcial.

###  Análise do Motor

Permite selecionar um motor individual e visualizar:

- RUL estimado;
- Health Score;
- Status;
- Priority Score;
- Classe de prioridade;
- RUL Score;
- Anomaly Score;
- Trend Score;
- interpretação operacional;
- composição do Health Score;
- evolução temporal dos indicadores;
- gráficos dos sensores;
- posição no ranking da frota;
- dados técnicos completos.

###  Modelo e Metodologia

Apresenta:

- arquitetura do pipeline;
- modelos utilizados;
- fórmulas;
- métricas;
- critérios de classificação;
- limitações do sistema.

---

##  Estrutura do projeto

```text
UPX 2.0/
│
├── app/
│   └── app.py
│
├── datasets/
│   └── CMAPSSData/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── modelo/
│   ├── fd001/
│   ├── fd001_capped125/
│   ├── fd001_anomalia/
│   └── fd001_final/
│
├── resultados/
│   ├── fd001_anomalia/
│   ├── fd001_tendencia/
│   └── fd001_pipeline_integrado/
│
├── treino/
│   ├── treino_fd001.py
│   └── treino_fd001_capped125.py
│
├── utils/
│   └── pipeline_fd001.py
│
├── requirements.txt
└── README.md
```

---

##  Requisitos

Recomendado:

```text
Python 3.11
```

Ambiente validado no desenvolvimento:

```text
Python 3.11.7
numpy 2.4.6
pandas 3.0.5
scikit-learn 1.7.2
joblib 1.6.0
tensorflow 2.21.0
tf-keras 2.21.0
streamlit 1.63.0
matplotlib 3.11.1
```

O Streamlit instala dependências adicionais usadas pela interface, incluindo Altair.

---

##  Como executar

### 1. Clonar o repositório

```bash
git clone URL_DO_SEU_REPOSITORIO
```

Entre na pasta:

```bash
cd "UPX 2.0"
```

### 2. Criar um ambiente virtual

#### Windows

```bash
py -3.11 -m venv .venv
```

Ativar:

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar arquivos necessários

Confirme a existência de:

```text
datasets/CMAPSSData/test_FD001.txt

modelo/fd001_capped125/modelo_rul_convertido.keras
modelo/fd001_capped125/scaler.pkl
modelo/fd001_capped125/config.json

modelo/fd001_anomalia/modelo_autoencoder_convertido.keras
modelo/fd001_anomalia/scaler.pkl
modelo/fd001_anomalia/config.json

modelo/fd001_final/config_pipeline.json

resultados/fd001_pipeline_integrado/resultados_100_motores.csv
resultados/fd001_pipeline_integrado/ranking_prioridade.csv
```

### 5. Executar o dashboard

Na raiz do projeto:

```bash
streamlit run app/app.py
```

No Windows também pode ser usado:

```bash
streamlit run app\app.py
```

O Streamlit normalmente abrirá o navegador automaticamente. Caso isso não aconteça, o terminal mostrará o endereço local, geralmente:

```text
http://localhost:8501
```

---

##  requirements.txt

```text
numpy==2.4.6
pandas==3.0.5
scikit-learn==1.7.2
joblib==1.6.0
tensorflow==2.21.0
tf-keras==2.21.0
streamlit==1.63.0
matplotlib==3.11.1
```

---

##  Análises parciais

RUL e anomalia precisam de pelo menos:

```text
30 ciclos
```

O Trend Score usa 10 janelas consecutivas. Por isso, uma análise completa precisa de aproximadamente:

```text
39 ciclos observados
```

Equipamentos com histórico menor podem ter RUL e Anomaly Score, mas não recebem Health Score nem Priority Score até existir histórico suficiente.

São marcados como:

```text
ANALISE_PARCIAL
```

---

##  Sensores utilizados

O pipeline utiliza 14 sensores do FD001:

```text
sensor_2
sensor_3
sensor_4
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

Entrada temporal principal:

```text
30 ciclos × 14 sensores
```

---

##  Dataset NASA C-MAPSS FD001

O FD001 é um dataset de simulação de degradação de motores turbofan.

No módulo atual:

- 100 trajetórias de treinamento;
- 100 trajetórias de teste;
- uma condição operacional principal;
- um modo de degradação relacionado ao sistema HPC;
- informações operacionais e 21 medições de sensores por observação.

O projeto utiliza uma seleção de 14 sensores para os modelos atuais.

---

##  Limitações

- O FD001 possui apenas um modo de degradação.
- O sistema não realiza diagnóstico universal de componentes.
- Os ciclos do benchmark não devem ser interpretados diretamente como horas ou dias.
- O modelo não deve ser utilizado diretamente em automóveis, trens, motores industriais ou outros ativos.
- Cada novo domínio exige dados, treinamento, validação e calibração próprios.
- Health Score e Priority Score são indicadores desenvolvidos para esta arquitetura.
- Estimativas de RUL possuem erro e incerteza.
- O projeto não substitui inspeção técnica ou sistemas certificados de manutenção.

---

##  Arquitetura modular

A proposta não é usar um único modelo para qualquer equipamento.

```text
Plataforma de Manutenção Preditiva
│
├── Módulo NASA C-MAPSS
│   ├── Modelo RUL
│   ├── Modelo de anomalia
│   ├── Trend Score
│   ├── Health Score
│   └── Priority Score
│
├── Futuro módulo automotivo
│   └── Telemetria OBD-II / ELM327
│
└── Futuros módulos industriais
    └── Modelos específicos por domínio
```

Cada domínio deve possuir seus próprios sensores, modelos e calibrações.

---

##  Possíveis extensões

- integração com telemetria real via OBD-II / ELM327;
- aquisição de dados de veículos reais;
- integração com banco de dados;
- dashboard de frota em tempo real;
- ordens de manutenção;
- simulação de custos e impacto operacional;
- novos datasets C-MAPSS;
- módulos ferroviários ou industriais;
- explicabilidade avançada;
- intervalo de confiança para RUL;
- implantação em nuvem.

---

##  Metodologia resumida

```text
FD001
 ↓
Parsing correto das colunas
 ↓
Seleção dos sensores
 ↓
Separação por motores
 ↓
Normalização
 ↓
Janelas temporais de 30 ciclos
 ↓
LSTM de RUL
 +
LSTM Autoencoder
 ↓
Anomaly Score
 ↓
Trend Score
 ↓
Health Score
 ↓
Status
 ↓
Priority Score
 ↓
Ranking operacional
 ↓
Dashboard Streamlit
```

---

##  Antes de enviar ao GitHub

Não envie o ambiente virtual para o repositório. Use um `.gitignore` como:

```gitignore
.venv/
venv/

__pycache__/
*.pyc
*.pyo

.streamlit/secrets.toml

.DS_Store
Thumbs.db

.vscode/
.idea/
```

Também confira o tamanho dos arquivos `.keras`. Caso algum arquivo seja grande demais para o GitHub, considere usar **Git LFS** ou armazenar os modelos separadamente.

---

##  Execução rápida no Windows

```bash
git clone https://github.com/MarceloDuraesLemos/ManutencaoPreditiva.git
cd "UPX 2.0"

py -3.11 -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

streamlit run app/app.py
```

---

##  Tecnologias

- Python
- TensorFlow / Keras
- LSTM
- LSTM Autoencoder
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Altair
- Matplotlib
- NASA C-MAPSS

---

##  Contexto acadêmico

Este projeto foi desenvolvido como uma plataforma experimental para estudo e demonstração de técnicas de:

- manutenção preditiva;
- prognóstico;
- detecção de anomalias;
- análise temporal;
- apoio à decisão de manutenção.

O foco atual é o módulo **NASA C-MAPSS FD001**.

---

##  UPX 2.0

**Plataforma modular de manutenção preditiva baseada em dados, Machine Learning e análise de degradação.**

# UPX 2.0 --- Plataforma de Manutenção Preditiva

A **UPX 2.0** é uma plataforma modular de manutenção preditiva
desenvolvida em Python, com interface em Streamlit e modelos de Machine
Learning aplicados a séries temporais de sensores.

A plataforma reúne dois módulos de análise --- **NASA C-MAPSS** e
**Automotivo** --- além de um **Normalizador Inteligente de
Telemetria**, responsável por interpretar e padronizar arquivos CSV
provenientes de diferentes fontes.

## Visão geral

A arquitetura foi organizada para que cada domínio possua seus próprios
dados, sensores, modelos e critérios de análise.

``` text
                         UPX 2.0
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
  NASA C-MAPSS                         Automotivo
          │                                   │
  Prognóstico de RUL                  Telemetria veicular
  Detecção de anomalias               Detecção de anomalias
  Tendência de degradação             Health Score
  Health Score                        Análise de sessões
  Prioridade de manutenção                    │
          │                                   │
          └──────────────┬────────────────────┘
                         │
                  Dashboard Streamlit
```

## Módulo NASA C-MAPSS

O módulo NASA utiliza o subconjunto **FD001 do C-MAPSS**, composto por
séries temporais de motores turbofan simulados.

O pipeline combina prognóstico e detecção de anomalias para acompanhar a
evolução da condição dos equipamentos.

``` text
Sensores do motor
       ↓
Pré-processamento
       ↓
Janelas temporais
       ↓
┌──────────────────┬────────────────────┐
│ LSTM para RUL    │ LSTM Autoencoder   │
└────────┬─────────┴─────────┬──────────┘
         │                   │
   RUL estimado        Anomaly Score
                             ↓
                        Trend Score
         └───────────┬───────┘
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

### RUL --- Remaining Useful Life

O modelo de prognóstico utiliza uma **LSTM** para estimar a vida útil
remanescente do motor em ciclos.

Configuração principal:

-   dataset: NASA C-MAPSS FD001;
-   janela temporal: 30 ciclos;
-   14 sensores selecionados;
-   target de treinamento com RUL limitado a 125 ciclos;
-   saída: RUL estimado em ciclos;
-   desempenho de teste: MAE de aproximadamente 11,14 ciclos e RMSE de
    aproximadamente 15,10 ciclos.

Os 14 sensores utilizados são:

``` text
sensor_2   sensor_3   sensor_4   sensor_7
sensor_8   sensor_9   sensor_11  sensor_12
sensor_13  sensor_14  sensor_15  sensor_17
sensor_20  sensor_21
```

### Detecção de anomalias

Um **LSTM Autoencoder** aprende o padrão de comportamento de referência
dos sensores. Durante a análise, o erro de reconstrução é convertido em
um **Anomaly Score de 0 a 100**.

Quanto maior o valor, maior o desvio observado em relação ao padrão
aprendido.

### Trend Score

O Trend Score acompanha a evolução recente do erro de reconstrução e
identifica se o comportamento anômalo está aumentando ao longo do tempo.

O cálculo utiliza 10 janelas consecutivas e uma regressão linear sobre a
evolução do erro.

### Health Score

O Health Score consolida os indicadores do módulo NASA em uma escala de
0 a 100:

``` text
Health Score =
0.60 × RUL Score
+ 0.25 × (100 - Anomaly Score)
+ 0.15 × (100 - Trend Score)
```

  Health Score   Condição
  -------------- ----------
  80--100        Saudável
  60--79,99      Atenção
  30--59,99      Risco
  0--29,99       Crítico

### Prioridade de manutenção

O Priority Score combina o risco associado ao Health Score e ao RUL:

``` text
Health Risk = 100 - Health Score
RUL Risk    = 100 - RUL Score

Priority Score =
0.30 × Health Risk
+ 0.70 × RUL Risk
```

  Priority Score   Prioridade
  ---------------- -----------------
  \< 20            P4 --- Baixa
  20 a \< 45       P3 --- Média
  45 a \< 75       P2 --- Alta
  ≥ 75             P1 --- Imediata

O dashboard permite visualizar a condição geral da frota, rankings de
prioridade e análises individuais dos motores.

------------------------------------------------------------------------

## Módulo Automotivo

O módulo Automotivo trabalha com séries temporais de telemetria veicular
e utiliza um **Autoencoder** treinado sobre dados de operação de
referência.

Diferentemente do módulo NASA, o módulo Automotivo **não estima RUL**.
Seu objetivo é identificar desvios multivariados no comportamento da
telemetria e apresentar indicadores de condição.

``` text
Telemetria automotiva
        ↓
Validação do contrato de entrada
        ↓
Janelas de 30 segundos
        ↓
Autoencoder
        ↓
Erro de reconstrução
        ↓
Anomaly Score
        ↓
Health Score
        ↓
Dashboard
```

O modelo trabalha com frequência esperada de **1 Hz**, janela temporal
de **30 amostras** e as seguintes oito variáveis:

  Feature             Grandeza
  ------------------- -----------------------------------------
  `rpm`               Rotação do motor
  `speed_kmh`         Velocidade em km/h
  `coolant_temp_c`    Temperatura do líquido de arrefecimento
  `engine_load_pct`   Carga do motor
  `throttle_pct`      Posição do acelerador
  `intake_temp_c`     Temperatura do ar de admissão
  `map_kpa`           Pressão absoluta do coletor
  `battery_voltage`   Tensão elétrica

A página Automotivo permite utilizar dados simulados ou importar sessões
em CSV. Cada arquivo é tratado como uma sessão independente para evitar
a criação de janelas artificiais entre viagens diferentes.

------------------------------------------------------------------------

## Normalizador Inteligente de Telemetria --- V7.5

Scanners automotivos e aplicativos de aquisição podem exportar a mesma
grandeza com nomes, unidades e frequências diferentes. O normalizador
funciona como uma camada entre esses arquivos e o contrato utilizado
pelo módulo Automotivo.

``` text
CSV bruto do scanner
        ↓
Limpeza estrutural
        ↓
Inventário dos sensores
        ↓
Identificação das grandezas físicas
        ↓
Mapeamento de aliases e unidades
        ↓
Comparação de fontes disponíveis
        ↓
Conversões de unidade
        ↓
Perfil de scanner / confirmação humana
        ↓
Regularização temporal conservadora em 1 Hz
        ↓
Validação das 8 features
        ↓
CSV padronizado para a UPX 2.0
```

### O que o normalizador faz

O normalizador é capaz de:

-   detectar e remover cabeçalhos repetidos no meio do arquivo;
-   identificar colunas temporais;
-   construir um inventário das colunas e sensores encontrados;
-   reconhecer aliases conhecidos;
-   classificar grandezas físicas;
-   validar unidades;
-   aplicar conversões conhecidas, como m/s para km/h;
-   comparar múltiplas fontes fisicamente compatíveis para a mesma
    feature;
-   solicitar confirmação quando existe uma alternativa válida que
    altera a fonte utilizada;
-   memorizar associações confirmadas em perfis de scanner;
-   reutilizar posteriormente uma associação salva, desde que ela
    continue fisicamente válida;
-   construir uma grade temporal conservadora de 1 Hz;
-   consolidar múltiplas medições reais do mesmo segundo pela mediana;
-   manter intervalos sem observação como ausentes;
-   verificar a disponibilidade das oito features exigidas pelo modelo;
-   calcular quantas janelas completas de 30 segundos podem ser
    utilizadas.

O normalizador **não preenche automaticamente sensores inexistentes**.
Se uma variável exigida pelo modelo não foi coletada, ela permanece
ausente e a inferência é bloqueada até existir uma janela válida.

### Perfis de scanner

Quando existem duas fontes válidas para a mesma variável, o sistema pode
apresentar a fonte atual e a alternativa tecnicamente preferida.

Exemplo:

``` text
Speed (GPS)(km/h)
        │
        ├── 94,63% de dados válidos
        │
        └── unidade nativa: km/h

GPS Speed (Meters/second)
        │
        ├── 100% de dados válidos
        │
        └── conversão: m/s × 3,6 → km/h
```

Após uma confirmação explícita, a associação pode ser armazenada em
`perfis_scanner.json`. Em arquivos futuros com o mesmo perfil
estrutural, a fonte confirmada pode ser reutilizada sem uma nova
pergunta, passando novamente pelas validações físicas antes de ser
aplicada.

### Arquivos gerados

Para um arquivo como:

``` text
viagem.csv
```

o normalizador pode gerar:

``` text
viagem_normalizado_upx.csv
viagem_regularizado_1hz_upx.csv
```

O arquivo `_normalizado_upx.csv` contém as features reconhecidas e
convertidas antes da regularização temporal.

O arquivo `_regularizado_1hz_upx.csv` contém a grade temporal de 1 Hz e
é o resultado preparado para a validação do contrato do módulo
Automotivo.

### Como executar o normalizador

Na raiz do projeto, com o ambiente virtual ativo:

``` bash
python ferramentas/normalizador_telemetria/normalizador_v7_trabalho.py
```

No Windows também pode ser executado com:

``` bat
.venv\Scripts\python.exe "ferramentas\normalizador_telemetria\normalizador_v7_trabalho.py"
```

O programa solicitará o caminho de um arquivo CSV ou de uma pasta
contendo arquivos CSV.

------------------------------------------------------------------------

## Fluxo completo de dados automotivos

O normalizador e o módulo Automotivo possuem responsabilidades
diferentes:

``` text
Scanner / aplicativo de telemetria
              ↓
          CSV bruto
              ↓
      Normalizador V7.5
              ↓
   CSV padronizado em 1 Hz
              ↓
   Validação das 8 features
              ↓
       ┌──────┴──────┐
       │             │
  Compatível     Incompleto
       │             │
       ↓             ↓
  Autoencoder    Diagnóstico das
       ↓         features ausentes
 Anomaly Score
       ↓
 Health Score
       ↓
   Dashboard
```

Assim, um CSV pode ser corretamente interpretado pelo normalizador e
ainda não possuir todos os sensores necessários para o modelo. Nesse
caso, o arquivo traduzido continua útil para inspeção e diagnóstico dos
dados disponíveis, mas não é enviado para inferência.

------------------------------------------------------------------------

## Dashboard Streamlit

A interface principal é executada em Streamlit e organiza os recursos em
módulos.

### NASA C-MAPSS

Apresenta visão geral dos motores, indicadores de condição, prioridades,
rankings, gráficos temporais e análise individual.

### Automotivo

Permite trabalhar com simulação ou importação de CSV, analisar sessões
independentes e visualizar os resultados produzidos pelo pipeline
automotivo.

------------------------------------------------------------------------

## Tecnologias utilizadas

-   **Python 3.11**
-   **Streamlit** --- interface e dashboards
-   **TensorFlow / Keras** --- redes neurais
-   **LSTM** --- prognóstico temporal de RUL
-   **LSTM Autoencoder / Autoencoder** --- detecção de anomalias
-   **scikit-learn** --- pré-processamento e métricas
-   **Pandas** --- manipulação e normalização dos dados
-   **NumPy** --- processamento numérico
-   **Matplotlib / Altair** --- visualização
-   **Joblib** --- persistência de componentes de pré-processamento
-   **PyInstaller** --- distribuição da aplicação para Windows
-   **NASA C-MAPSS FD001** --- base utilizada pelo módulo de prognóstico

------------------------------------------------------------------------

## Estrutura principal do projeto

``` text
UPX 2.0/
│
├── app/
│   ├── app.py
│   └── paginas/
│       ├── 01_nasa_cmapss.py
│       └── 02_automotivo.py
│
├── datasets/
│   └── CMAPSSData/
│
├── ferramentas/
│   └── normalizador_telemetria/
│       ├── normalizador_v7_trabalho.py
│       ├── sensores.json
│       └── perfis_scanner.json
│
├── modelo/
│   ├── fd001_capped125/
│   ├── fd001_anomalia/
│   ├── fd001_final/
│   └── automotivo_anomalia/
│
├── resultados/
│   └── fd001_pipeline_integrado/
│
├── utils/
│   ├── pipeline_fd001.py
│   └── pipeline_automotivo.py
│
├── distribuicao/
│   └── launcher.py
│
├── Iniciar_Plataforma.bat
├── requirements.txt
├── UPX_2_0.spec
└── README.md
```

------------------------------------------------------------------------

## Como executar o projeto

### Opção rápida no Windows

Com o repositório já configurado, execute:

``` text
Iniciar_Plataforma.bat
```

O script utiliza o Python do ambiente virtual e inicia a interface
Streamlit.

### Execução manual

Clone o repositório:

``` bash
git clone https://github.com/MarceloDuraesLemos/ManutencaoPreditiva.git
cd ManutencaoPreditiva
```

Crie o ambiente virtual:

``` bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Instale as dependências:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Inicie a plataforma:

``` bash
python -m streamlit run app/app.py
```

O navegador será aberto com a interface da UPX 2.0.

------------------------------------------------------------------------

## Utilização

### NASA C-MAPSS

1.  Abra o módulo **NASA C-MAPSS**.
2.  Consulte o dashboard geral ou selecione um motor.
3.  Analise RUL, condição, anomalias, tendência e prioridade.
4.  Utilize os gráficos e dados técnicos para acompanhar a evolução
    temporal.

### Automotivo com dados compatíveis

1.  Abra o módulo **Automotivo**.
2.  Selecione **Simulação** ou **Importar CSV**.
3.  Para CSV, utilize um arquivo que contenha as oito features exigidas
    pelo modelo.
4.  Execute a análise.
5.  Consulte os indicadores e gráficos da sessão.

### Automotivo com CSV de scanner externo

``` text
CSV do scanner
      ↓
Normalizador V7.5
      ↓
*_regularizado_1hz_upx.csv
      ↓
Módulo Automotivo
```

1.  Execute o normalizador.
2.  Informe o CSV bruto ou uma pasta de CSVs.
3.  Revise eventuais decisões de fonte apresentadas.
4.  Utilize o arquivo `_regularizado_1hz_upx.csv` gerado.
5.  Importe o arquivo no módulo Automotivo.
6.  Se todas as oito features e uma janela válida estiverem disponíveis,
    o arquivo poderá seguir para o modelo.
7.  Se faltarem sensores, a análise é interrompida sem criar medições
    artificiais.

------------------------------------------------------------------------

## Distribuição para Windows

O projeto possui configuração de empacotamento com **PyInstaller**.

A distribuição é gerada no formato `onedir`, mantendo o executável e
suas dependências no mesmo diretório:

``` text
dist/
└── UPX_2_0/
    ├── UPX_2_0.exe
    └── _internal/
```

Para utilizar essa distribuição, deve ser mantida a pasta `UPX_2_0`
completa. O executável não deve ser separado do diretório `_internal`.

------------------------------------------------------------------------

## Arquitetura modular

A UPX 2.0 foi estruturada para permitir que diferentes domínios utilizem
modelos específicos:

``` text
UPX 2.0
│
├── NASA C-MAPSS
│   ├── RUL
│   ├── Anomalias
│   ├── Tendência
│   ├── Health Score
│   └── Prioridade
│
├── Automotivo
│   ├── Telemetria
│   ├── Autoencoder
│   ├── Anomalias
│   └── Health Score
│
└── Normalizador de Telemetria
    ├── Inventário de sensores
    ├── Mapeamento físico
    ├── Conversão de unidades
    ├── Perfis de scanner
    └── Regularização temporal
```

Novos domínios podem ser adicionados mantendo modelos, sensores e
critérios próprios, sem reutilizar automaticamente modelos treinados
para equipamentos diferentes.

------------------------------------------------------------------------

## UPX 2.0

**Plataforma modular de manutenção preditiva baseada em séries
temporais, Machine Learning, prognóstico e detecção de anomalias.**

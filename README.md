# Delivery ETA Prediction — DoorDash

Projeto de Machine Learning para previsão do tempo total de entrega de pedidos de delivery, utilizando o dataset `historical_data.csv` disponibilizado pela StrataScratch no exercício **DoorDash Delivery Duration Prediction**.

## Objetivo

Desenvolver uma solução ponta a ponta com boas práticas de Engenharia de Dados, Análise Exploratória (EDA), Modelagem Preditiva, Avaliação e Documentação para estimar a duração total de entrega de um pedido.

## Estrutura do Projeto

```text
01-delivery-eta/
├── app/                  # Aplicação para demonstração/inferência
├── data/
│   ├── raw/              # Dados brutos originais (imutáveis, ignorados pelo Git)
│   └── processed/        # Dados tratados e prontos para modelagem (ignorados pelo Git)
├── models/               # Artefatos de modelos treinados
├── notebooks/            # Notebooks de auditoria, EDA e experimentação
├── reports/
│   └── figures/          # Gráficos e relatórios gerados
├── src/                  # Código-fonte modularizado (processamento, treino, avaliação)
├── .gitignore
├── README.md
└── requirements.txt
```

## Configuração do Ambiente

Compatível com **VS Code** (ambiente virtual local) e **Google Colab**.

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Status Atual

- [x] Estrutura inicial do projeto criada
- [ ] Fornecimento e auditoria do dataset `historical_data.csv`
- [ ] Análise Exploratória de Dados (EDA)
- [ ] Engenharia de atributos e pré-processamento
- [ ] Treinamento e avaliação de modelos
- [ ] Documentação final e aplicação

# DoorDash Delivery ETA Prediction — Documento Consolidado de Resultados

Este documento resume, em ordem linear, todas as decisões metodológicas, os principais resultados obtidos, os pontos de atenção e um guia prático de leitura para navegar por todo o projeto.

---

## 1. Sumário: Como Ler e Entender Todo o Projeto

Se você está abrindo este repositório pela primeira vez, siga esta ordem de leitura:

### Visão Geral e Código Reutilizável (`src/`)
1. [README.md](README.md) — Objetivo, estrutura de diretórios e configuração do ambiente.
2. [src/data_preparation.py](src/data_preparation.py) — Leitura imutável do bruto, tipagem, cálculo do alvo e tratamento controlado de inválidos.
3. [src/feature_engineering.py](src/feature_engineering.py) — Criação de atributos e separação entre `CONFIRMED_FEATURES`, `UNCERTAIN_EXPERIMENTAL_FEATURES` e `FORBIDDEN_LEAKAGE_COLS`.
4. [src/temporal_validation.py](src/temporal_validation.py) — Divisão temporal com embargo baseado em `actual_delivery_time`.
5. [src/model_training.py](src/model_training.py) — Pré-processamento sem *leakage*, treino, métricas e ablações.

### Ordem Linear de Análise e Relatórios (`notebooks/` e `reports/`)
| Etapa | Foco | Notebook | Relatório |
| :---: | :--- | :--- | :--- |
| **01** | Auditoria profunda de qualidade | [01_data_quality_investigation.ipynb](notebooks/01_data_quality_investigation.ipynb) | [data_quality_investigation.md](reports/data_quality_investigation.md) |
| **02** | Preparação controlada dos dados | [02_data_preparation.ipynb](notebooks/02_data_preparation.ipynb) | [data_preparation_report.md](reports/data_preparation_report.md) |
| **03** | Análise Exploratória (EDA) | [03_exploratory_data_analysis.ipynb](notebooks/03_exploratory_data_analysis.ipynb) | [eda_report.md](reports/eda_report.md) |
| **04** | Engenharia de atributos | [04_feature_engineering.ipynb](notebooks/04_feature_engineering.ipynb) | [feature_engineering_report.md](reports/feature_engineering_report.md) |
| **05** | Auditoria de features e divisão temporal | [05_temporal_validation.ipynb](notebooks/05_temporal_validation.ipynb) | [temporal_validation_report.md](reports/temporal_validation_report.md) |
| **06** | Baseline, modelos iniciais e Colab | [06_model_training.ipynb](notebooks/06_model_training.ipynb) / [Colab](notebooks/06_model_training_colab.ipynb) | [model_training_report.md](reports/model_training_report.md) |
| **06.1** | Outliers e otimização de perda | — | [model_optimization_report.md](reports/model_optimization_report.md) |
| **06.2** | Auditoria final e congelamento | — | [final_model_audit.md](reports/final_model_audit.md) |
| **07** | Avaliação final no conjunto de teste | [07_final_test_evaluation.ipynb](notebooks/07_final_test_evaluation.ipynb) | [final_test_evaluation_report.md](reports/final_test_evaluation_report.md) |
| **08** | Interpretabilidade e investigação de erros | [08_model_interpretability.ipynb](notebooks/08_model_interpretability.ipynb) | [model_interpretability_report.md](reports/model_interpretability_report.md) |

*(Artefatos finais adicionais: [final_test_metrics.json](reports/final_test_metrics.json), [feature_importance.csv](reports/feature_importance.csv) e o pipeline salvo em `models/final_hgb_eta_pipeline.joblib`.)*

---

## 2. Linha do Tempo Linear das Principais Decisões

1. **Auditoria Inicial e Profunda (Etapa 01):**
   - Dataset `historical_data.csv` possui `197.428` linhas × `16` colunas (`0` duplicatas exatas).
   - Identificados `7` registros sem `actual_delivery_time`, `3` outliers extremos `> 24h` (máximo `~98,6 dias` em um único registro de `2014-10-19`), valores negativos em `4` colunas (`min_item_price`, `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders`), ausência simultânea dos `3` campos operacionais em `16.262` linhas (`95,47%` no `market_id = 6`) e `total_busy_dashers > total_onshift_dashers` em `22,30%` dos válidos (mantido como válido).
2. **Preparação Controlada (Etapa 02):**
   - Arquivo bruto preservado intacto. Criadas três visões: `df_raw` (`197.428`), `df_analysis` (`197.428`, com negativos preservados) e `df_model` (`197.421`, removendo **apenas** os `7` registros sem alvo; `0` com duração `<= 0`).
   - Negativos nas 4 features convertidos para `NaN` apenas em `df_model`, sem imputação e sem remoção de outliers.
3. **Análise Exploratória (Etapa 03):**
   - Mediana da duração = `44,33 min`; `80,14%` dos pedidos entregues em até `60 min`; `1.090` (`0,55%`) acima de `120 min`.
   - Correlação de Spearman revelou relações monotônicas consistentes em `estimated_store_to_consumer_driving_duration` (`0,2659`), `subtotal` (`0,2354`) e `total_outstanding_orders` (`0,1800`), enquanto Pearson ficou próximo de zero devido aos outliers `> 24h`.
4. **Feature Engineering e Controle de Vazamento (Etapa 04):**
   - Horários mantidos no timestamp registrado (aparente UTC, sem offset local disponível).
   - Criadas features de pedido (`average_item_price`, `item_price_range`, `distinct_item_ratio`, `is_single_item`, `is_large_order`), razões operacionais sem teto de 100% e `historical_store_order_count` (estritamente causal, sem usar o alvo).
   - Separadas features **confirmadas** no momento do pedido de features **experimentais/incertas** (operacionais e `driving_duration`).
5. **Auditoria de Features e Validação Temporal (Etapa 05):**
   - Corrigidos `325` empates de `(store_id, created_at)` em `historical_store_order_count` e criada a flag `invalid_item_price_range` (`800` casos onde `max_item_price < min_item_price`).
   - Definidos cortes temporais às `10:00:00` UTC (`2015-02-10` e `2015-02-14`, janela sem pedidos entre `09h–13h` UTC), resultando em **`0` cruzamentos** com embargo por `actual_delivery_time`: **Treino `138.572`**, **Validação `28.808`**, **Teste `30.041`** (isolado).
6. **Modelagem Inicial, Otimização de Perda e Congelamento (Etapas 06, 06.1 e 06.2):**
   - `store_id` (`6.743` IDs) foi excluído explicitamente do One-Hot Encoding para evitar explosão dimensional, resultando em `21` features confirmadas (`CONFIRMED_MODEL_FEATURES`).
   - Sob perda quadrática (`squared_error`), os `2` outliers `> 24h` do treino distorceram severamente a `Random Forest` (RMSE = `57,71 min`, previsões de até `5.126 min`).
   - A troca para `loss="absolute_error"` no `HistGradientBoostingRegressor` resolveu a distorção sem remover outliers.
   - A auditoria de `early_stopping='auto'` identificou uso de split interno **aleatório**; substituído por `early_stopping=False` com `max_iter=20` (definido por split interno temporal 90/10 dentro do treino), congelando o modelo final.
7. **Avaliação Final no Teste e Interpretabilidade (Etapas 07 e 08):**
   - Executada avaliação única no teste (`n = 30.041`) sem alterações no modelo, seguida de *Permutation Importance*, dependência parcial e investigação detalhada de erros na validação.

---

## 3. Principais Resultados

### 3.1 Desempenho do Modelo Congelado vs Baseline (`21` Features Confirmadas)
Modelo congelado: `HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.08, max_iter=20, max_leaf_nodes=63, min_samples_leaf=30, early_stopping=False, random_state=42)`.

| Conjunto | Modelo | MAE (min) | RMSE (min) | MedAE (min) | P90_AE (min) | R² | Mean Bias |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Validação** (`n=28.808`) | Baseline (Median) | 12,8201 | 18,6790 | 9,9000 | 25,6667 | -0,0341 | -3,3927 |
| **Validação** (`n=28.808`) | **HGB Congelado** | **12,1142** | **17,9219** | **9,2317** | **24,7196** | **+0,0480** | -2,8943 |
| **Teste** (`n=30.041`) | Baseline (Median) | 14,1955 | 38,1036 | 10,4667 | 29,0833 | -0,0195 | -5,2643 |
| **Teste** (`n=30.041`) | **HGB Congelado** | **12,7797** | **37,1903** | **9,2282** | **25,9125** | **+0,0288** | -4,6494 |

- **Ganho no teste vs baseline:** `-9,97%` em MAE (`-1,42 min`), `-11,83%` em MedAE (`-1,24 min`), `-10,90%` em P90_AE (`-3,17 min`) e `-2,40%` em RMSE (`-0,91 min`).
- **Estabilidade:** `MedAE` praticamente idêntica entre validação (`9,23 min`) e teste (`9,23 min`).

### 3.2 Ablações Experimentais (Validação — Disponibilidade Incerta)
Testadas separadamente na Etapa 06.1 com `HGB (absolute_error, early_stopping='auto')`:
- **Confirmadas (`21`):** MAE `12,17 min` | RMSE `18,06 min` | R² `+0,0338`
- **+ Estimativas Driving (`23`):** MAE `11,65 min` | RMSE `17,53 min` | R² `+0,0895`
- **+ Razões Operacionais (`23`):** **MAE `11,09 min`** | **RMSE `16,90 min`** | **R² `+0,1538`**

### 3.3 Variáveis Mais Relevantes (Permutation Importance na Validação)
1. `created_hour` (`+0,772 min` no MAE)
2. `subtotal` (`+0,415 min`)
3. `market_id` (`+0,131 min`)
4. `created_dayofweek` (`+0,092 min`)
5. `estimated_order_place_duration` (`+0,090 min`)

---

## 4. Pontos de Atenção e Limitações

1. **Não está pronto para produção real:** Com `R² = +0,0288` no teste, variáveis estáticas de checkout explicam apenas uma pequena fração da variância total da entrega.
2. **Subestimação de atrasos e superestimação de entregas rápidas:** Na validação (`<= 30 min`: Mean Bias `+16,16 min`; `> 120 min`: Mean Bias `-124,46 min`) e no teste (`<= 30 min`: `+15,62 min`; `> 120 min`: `-135,20 min`), o modelo prevê próximo à região central (`36–60 min`) porque não observa congestionamento nem distância em tempo real.
3. **Outliers e qualidade de dados:** Presença de `3` registros `> 24h` no dataset (um deles em `2014-10-19`, com `~98,6 dias`), `800` casos de `max_item_price < min_item_price` e agrupamento de fechamentos próximos às `07:00` UTC em `2015-02-12`.
4. **Fuso horário e janela curta:** Dados concentrados em ~1 mês (`2015-01-21` a `2015-02-18`) em provável horário UTC, sem mapeamento de fuso local por mercado.
5. **Próximos passos para uma v2:** Confirmar disponibilidade em tempo real das razões operacionais e `driving_duration`, adicionar *target encoding* temporal causal por `store_id` e avaliar modelagem quantílica (ex.: P50 e P90).

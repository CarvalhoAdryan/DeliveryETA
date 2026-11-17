# DoorDash Delivery ETA — Relatório de Preparação Controlada dos Dados

**Código reutilizável:** [data_preparation.py](../src/data_preparation.py)  
**Notebook:** [02_data_preparation.ipynb](../notebooks/02_data_preparation.ipynb)  
**Arquivo original (imutável):** `data/raw/historical_data.csv`

---

## 1. Resumo das Decisões Aplicadas

1. **Preservação do original e separação em 3 DataFrames:**
   - `df_raw`: leitura fiel do CSV original (`197.428` × `16`).
   - `df_analysis`: cópia com tipos corrigidos, variável-alvo, flags de validade e atributos iniciais (`197.428` × `30`), **sem exclusão de registros** e **preservando valores negativos**.
   - `df_model`: cópia para modelagem inicial (`197.421` × `30`).
2. **Conversão de tipos:**
   - `created_at` e `actual_delivery_time` convertidos para `datetime64[ns]`.
   - `market_id` e `order_protocol` convertidos para `category` (via `Int64` para preservar identificadores inteiros sem sufixo `.0` e manter `NaN`).
3. **Variável-alvo sem arredondamento:**
   - `delivery_duration_seconds = (actual_delivery_time - created_at).dt.total_seconds()`
   - `delivery_duration_minutes = delivery_duration_seconds / 60`
4. **Filtro de validade do alvo:**
   - Removidos **apenas `7` registros** (`0,0035%`) com alvo ausente (`actual_delivery_time` nulo).
   - `0` registros com duração `<= 0`.
   - Outliers plausíveis/extremos (`> 120m`, `> 180m`, `> 24h`) foram sinalizados por flags, mas **mantidos** sem clipping ou limite máximo arbitrário.
5. **Valores negativos nas features (`df_model`):**
   - Convertidos para `NaN` apenas em `min_item_price`, `total_onshift_dashers`, `total_busy_dashers` e `total_outstanding_orders`, **sem imputação**.
6. **Engenharia de atributos inicial:**
   - Temporais: `created_hour`, `created_dayofweek`, `created_month`, `created_is_weekend`.
   - Operacionais: `busy_greater_than_onshift` (tipo `boolean` nullable, retornando `<NA>` quando algum dos dois campos está ausente) e `dasher_count_difference = total_onshift_dashers - total_busy_dashers` (preservando valores negativos).

---

## 2. Dimensões e Registros Removidos

| Versão | Linhas | Colunas | Observação |
| :--- | ---: | ---: | :--- |
| `df_raw` | 197.428 | 16 | Dados brutos originais |
| `df_analysis` | 197.428 | 30 | Sem exclusões; negativos preservados |
| `df_model` | 197.421 | 30 | Removidos `7` registros com alvo nulo (`0` com alvo `<= 0`) |

---

## 3. Contabilização dos Alvos (`df_analysis`)

| Condição | Quantidade | % sobre Total |
| :--- | ---: | ---: |
| Total de registros | 197.428 | 100,0000% |
| Alvo ausente (`NaN`) | 7 | 0,0035% |
| Alvo `<= 0` | 0 | 0,0000% |
| Alvo válido (`> 0`) | 197.421 | 99,9965% |
| Duração `> 120 min` | 1.090 | 0,5521% |
| Duração `> 180 min` | 138 | 0,0699% |
| Duração `> 24 horas` | 3 | 0,0015% |

---

## 4. Valores Negativos Convertidos em `NaN` (`df_model`)

| Coluna | Negativos em `df_analysis` | Convertidos para `NaN` em `df_model` |
| :--- | ---: | ---: |
| `min_item_price` | 13 | 13 |
| `total_onshift_dashers` | 21 | 21 |
| `total_busy_dashers` | 21 | 21 |
| `total_outstanding_orders` | 44 | 44 |

---

## 5. Valores Ausentes em `df_model`

| Coluna | Ausentes (`df_model`) | % (`df_model`) |
| :--- | ---: | ---: |
| `total_outstanding_orders` | 16.306 | 8,2595% |
| `busy_greater_than_onshift` | 16.303 | 8,2580% |
| `dasher_count_difference` | 16.303 | 8,2580% |
| `total_onshift_dashers` | 16.283 | 8,2479% |
| `total_busy_dashers` | 16.283 | 8,2479% |
| `store_primary_category` | 4.760 | 2,4111% |
| `order_protocol` | 995 | 0,5040% |
| `market_id` | 987 | 0,4999% |
| `estimated_store_to_consumer_driving_duration` | 526 | 0,2664% |
| `min_item_price` | 13 | 0,0066% |

*(Nota: o aumento de ausências em `busy_greater_than_onshift` e `dasher_count_difference` em relação a `16.262` decorre da conversão dos valores negativos de `total_onshift_dashers` e `total_busy_dashers` para `NaN` em `df_model`.)*

---

## 6. Comparação da Distribuição do Alvo (Antes vs Depois dos Filtros de Validade)

Como os únicos `7` registros excluídos possuíam `actual_delivery_time` nulo (e não havia nenhum valor `<= 0`), a distribuição dos valores válidos permanece idêntica entre `df_analysis` (excluindo nulos) e `df_model`:

| Estatística | Segundos (`df_analysis` válido / `df_model`) | Minutos (`df_analysis` válido / `df_model`) |
| :--- | ---: | ---: |
| Contagem | 197.421 | 197.421 |
| Média | 2.908,26 | 48,47 |
| Desvio Padrão | 19.229,61 | 320,49 |
| Mínimo | 101,00 | 1,68 |
| 25% | 2.104,00 | 35,07 |
| 50% (Mediana) | 2.660,00 | 44,33 |
| 75% | 3.381,00 | 56,35 |
| Máximo | 8.516.859,00 | 141.947,65 |

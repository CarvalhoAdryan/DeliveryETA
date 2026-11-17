# DoorDash Delivery ETA — Relatório de Feature Engineering (Etapa 04)

**Módulo reutilizável:** [feature_engineering.py](../src/feature_engineering.py)  
**Notebook:** [04_feature_engineering.ipynb](../notebooks/04_feature_engineering.ipynb)  
**Base de entrada:** `df_model` (`197.421` linhas × `30` colunas) → **Saída:** `df_feat` (`197.421` linhas × `41` colunas)

---

## 1. Investigação de Fuso Horário (`created_at`)

- Em todos os `6` mercados (`market_id`), o pico de pedidos ocorre entre `01h` e `03h` (com `02h` sendo a hora #1 em todos eles), e as horas `[9, 10, 11, 12, 13]` possuem **zero pedidos** em todo o dataset.
- **Interpretação e limitação:** Esse padrão é compatível com timestamps armazenados em **UTC** para mercados norte-americanos (ex.: `02h UTC` ≈ `18h PST`, horário de jantar; `09h–13h UTC` ≈ `01h–05h PST`, madrugada). Como o dataset não informa fuso horário ou cidade por `market_id`, **não é possível confirmar o horário local exato**. Por isso, mantivemos `created_hour`, `created_dayofweek`, `created_is_weekend` e `created_time_block` (`00-05h_reg`, `06-11h_reg`, `12-17h_reg`, `18-23h_reg`) sobre o **horário registrado**.

---

## 2. Catálogo de Features Criadas

| Feature | Fórmula / Regra | Justificativa (EDA) | Ausentes (`n` / %) | Distribuição Observada | Disponibilidade & Risco de Vazamento | Recomendação |
| :--- | :--- | :--- | ---: | :--- | :--- | :--- |
| `created_hour` | `created_at.dt.hour` | Variação horária de mediana (`36,68` a `51,22 min`) | 0 (0,00%) | `min=0`, `p50=3`, `max=23` | **Confirmada** / Baixo (fuso UTC provável) | **Usar** |
| `created_dayofweek` | `created_at.dt.dayofweek` | Sábado (`47,10 min`) vs Quarta (`41,58 min`) | 0 (0,00%) | `0` a `6` (`p50=3`) | **Confirmada** / Baixo | **Usar** |
| `created_is_weekend` | `created_dayofweek >= 5` | Fim de semana (`46,20`) > útil (`43,38 min`) | 0 (0,00%) | Média `0,35` (35% FDS) | **Confirmada** / Baixo | **Usar** |
| `created_time_block` | Faixas de 6h do horário registrado | Captura blocos sem assumir hora local | 0 (0,00%) | `00-05h`: 127.243; `18-23h`: 62.649 | **Confirmada** / Baixo | **Usar** |
| `average_item_price` | `subtotal / total_items` (`total_items > 0`) | Separa ticket alto por quantidade vs preço unitário | 0 (0,00%) | `p50=895`, `p95=1.850`, `max=14.700` | **Confirmada** / Baixo | **Usar** |
| `item_price_range` | `max_item_price - min_item_price` | Dispersão de preços no carrinho | 13 (0,01%) | `min=-198`*, `p50=370`, `p95=1.373`, `max=8.400` | **Confirmada** / Baixo | **Usar** |
| `distinct_item_ratio` | `num_distinct_items / total_items` | Diversidade de preparo vs itens repetidos | 0 (0,00%) | `min=0,01`, `p50=1,00`, `mean=0,90` | **Confirmada** / Baixo | **Usar** |
| `is_single_item` | `total_items == 1` | Pedidos unitários têm menor mediana (`41,18 min`) | 0 (0,00%) | Média `0,21` (21%) | **Confirmada** / Baixo | **Usar** |
| `is_large_order` | `total_items >= 5` | Faixas `5-8` (`48,02`) e `9+` (`51,92 min`) | 0 (0,00%) | Média `0,19` (19%) | **Confirmada** / Baixo | **Usar** |
| `historical_store_order_count` | `cumcount()` por `store_id` ordenado por `created_at` | Volume acumulado anterior da loja (sem usar o alvo) | 0 (0,00%) | `min=0`, `p50=30`, `p95=271`, `max=936` | **Confirmada** / Baixo (estritamente causal) | **Usar** |
| `dasher_count_difference` | `total_onshift_dashers - total_busy_dashers` | Pico de mediana próximo a `0` (`48,38 min`) | 16.303 (8,26%) | `min=-64`, `p50=1`, `p95=22`, `max=86` | **Incerta** / Médio (snapshot operacional) | **Ablação** |
| `busy_greater_than_onshift` | `busy > onshift` (quando ambos válidos) | Ocorre em `22,3%` dos válidos; mediana `+1,37 min` | 16.303 (8,26%) | `False`: 140.745; `True`: 40.373 | **Incerta** / Médio | **Ablação** |
| `is_operational_missing` | Qualquer dos 3 campos operacionais é `NaN` | `95,47%` ausente no `market_id = 6` | 0 (0,00%) | Média `0,08` (8,26%) | **Incerta** / Médio | **Ablação** |
| `outstanding_orders_per_onshift_dasher` | `total_outstanding_orders / total_onshift_dashers` (`onshift > 0`) | Mede pressão de demanda por entregador | 19.939 (10,10%) | `p50=1,20`, `p95=1,90`, `max=47,00` | **Incerta** / Médio | **Ablação** |
| `busy_dashers_ratio` | `total_busy_dashers / total_onshift_dashers` (`onshift > 0`) | Taxa de ocupação sem teto de 100% | 19.915 (10,09%) | `p50=0,96`, `p95=1,33`, `max=31,00` | **Incerta** / Médio | **Ablação** |
| `estimated_total_duration` | `estimated_order_place_duration + estimated_store_to_consumer_driving_duration` | Baseline aditivo experimental das estimativas | 526 (0,27%) | `min=153`, `p50=850`, `p95=1.247`, `max=3.222` | **Incerta** / Médio (`driving_duration` incerta) | **Ablação** |

*\*Nota: `item_price_range` apresenta valor mínimo `-198` em casos raros onde `max_item_price < min_item_price` nos dados brutos.*

---

## 3. Estratégia para Features Históricas Baseadas no Alvo (Loja e Categoria)

Não calculamos `store_historical_median_duration` nem `category_historical_median_duration` nesta etapa porque:
1. Ordenar apenas por `created_at` e acumular `delivery_duration_minutes` pode causar **vazamento futuro**, pois um pedido criado antes (`created_at_j < created_at_i`) pode ainda estar em trânsito (`actual_delivery_time_j > created_at_i`) no momento da previsão.
2. **Estratégia segura para a etapa de modelagem/validação:** Calcular estatísticas históricas (ex.: mediana suavizada por loja e por categoria) **estritamente dentro do conjunto de treino temporal** (considerando apenas pedidos com `actual_delivery_time <= t_cutoff`), ou por *out-of-fold* temporal respeitando `actual_delivery_time < created_at` do bloco de validação.

---

## 4. Separação dos Conjuntos para Modelagem

- **`CONFIRMED_FEATURES` (`21` variáveis):** Conjunto conservador, livre de vazamento e com disponibilidade confirmada no momento do pedido.
- **`UNCERTAIN_EXPERIMENTAL_FEATURES` (`10` variáveis):** Variáveis operacionais e estimativas de deslocamento mantidas separadas para **experimentos controlados de ablação** na etapa de modelagem (comparando modelo apenas com features confirmadas vs modelo com features experimentais).
- **`FORBIDDEN_LEAKAGE_COLS`:** `actual_delivery_time`, `delivery_duration_seconds`, `delivery_duration_minutes` e todas as flags derivadas do alvo.

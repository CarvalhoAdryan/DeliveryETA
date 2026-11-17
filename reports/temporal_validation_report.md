# DoorDash Delivery ETA — Relatório de Auditoria de Features e Validação Temporal (Etapa 05)

**Módulos:** [feature_engineering.py](../src/feature_engineering.py) | [temporal_validation.py](../src/temporal_validation.py)  
**Notebook:** [05_temporal_validation.ipynb](../notebooks/05_temporal_validation.ipynb)

---

## 1. Auditoria das Features (`df_feat`)

- **Dimensões:** `197.421` linhas × `42` colunas (após inclusão de `invalid_item_price_range`: `16` originais + `14` derivadas de `data_preparation.py` + `12` derivadas de `feature_engineering.py`).
- **Classificação completa:**
  - `CONFIRMED_FEATURES`: `22` variáveis (incluindo `invalid_item_price_range`).
  - `UNCERTAIN_EXPERIMENTAL_FEATURES`: `10` variáveis.
  - `FORBIDDEN_LEAKAGE_COLS`: `9` variáveis (`actual_delivery_time`, `delivery_duration_seconds`, `delivery_duration_minutes` e `6` flags do alvo).
  - `created_at`: `1` coluna de timestamp de referência.
  - **Total verificado:** `22 + 10 + 9 + 1 = 42` colunas, com **zero** sobreposição entre features permitidas e colunas proibidas.

---

## 2. Auditoria de `historical_store_order_count`

- **Problema identificado:** Existiam **`325` registros** com exatamente o mesmo par `(store_id, created_at)`. O uso direto de `cumcount()` atribuía `0, 1, 2...` a pedidos simultâneos, fazendo com que um pedido contasse outro pedido do mesmo segundo como histórico.
- **Correção aplicada:** Em [feature_engineering.py](../src/feature_engineering.py), os dados são agrupados por `(store_id, created_at)` para somar as ocorrências por timestamp e subtrair a contagem do próprio timestamp (`cumsum() - _ts_count`). Verificado que o máximo de valores distintos por empate agora é **`1`**.

---

## 3. Auditoria de `item_price_range`

- **Quantidade de valores negativos:** **`800` registros** (`0,41%` de `197.421`) possuem `max_item_price < min_item_price` (valores entre `-198` e `-1`).
- **Padrão observado:** Diversos registros têm `total_items == 1` com `min_item_price > max_item_price` (ex.: `min=1314`, `max=1292`, `range=-22`), indicando inconsistência nos campos de preço unitário bruto.
- **Decisão:** Preservamos os valores negativos originais em `item_price_range` (sem substituir por zero) e adicionamos a flag binária `invalid_item_price_range` (`1` quando `item_price_range < 0`).

---

## 4. Auditoria das Razões Operacionais

| Razão | `n` válidos | `> 5` | `> 10` | P99 | Máximo | Causa dos Extremos |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `outstanding_orders_per_onshift_dasher` | 177.482 | 60 | 12 | 2,33 | 47,00 | Denominador pequeno (`total_onshift_dashers = 1` ou `2`) |
| `busy_dashers_ratio` | 177.506 | 156 | 42 | 2,00 | 31,00 | Denominador pequeno (`total_onshift_dashers = 1`) |

- Os valores originais foram preservados sem clipping nesta etapa.

---

## 5. Validação Temporal: Simples vs Embargo

Para evitar vazamento temporal entre períodos (pedidos criados antes do corte cuja entrega `actual_delivery_time` só termina após o corte), comparamos dois pontos de corte:

### 5.1 Corte à Meia-Noite (`00:00:00` UTC: `2015-02-10` e `2015-02-14`)
- **Cruzamentos descartados pelo embargo:** `161` pedidos (Treino → Validação) + `197` pedidos (Validação → Teste) = **`358` registros**.
- Como `00:00 UTC` coincide com horário ativo de pedidos nos mercados, centenas de entregas ficam em trânsito na virada do corte.

### 5.2 Corte Recomendado na Janela Sem Pedidos (`10:00:00` UTC: `2015-02-10` e `2015-02-14`)
Como a EDA mostrou **zero pedidos criados entre `09h` e `13h` UTC**, posicionar os cortes às `10:00:00` UTC elimina naturalmente os cruzamentos de janela:

| Conjunto | `n` | `created_at` (mín – máx) | `actual_delivery_time` (mín – máx) | P50 (min) | Média (min) | P90 (min) | Mercados | Categorias |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Train** | 138.572 (70,2%) | `2014-10-19 05:24` – `2015-02-10 05:59` | `2015-01-21 15:58` – `2015-02-10 06:55` | 44,02 | 48,52 | 70,47 | 6 | 74 |
| **Val** | 28.808 (14,6%) | `2015-02-10 14:37` – `2015-02-14 07:00` | `2015-02-10 15:34` – `2015-02-14 08:14` | 44,65 | 47,41 | 68,73 | 6 | 70 |
| **Test** | 30.041 (15,2%) | `2015-02-14 14:37` – `2015-02-18 06:00` | `2015-02-14 15:49` – `2015-02-19 22:45` | 45,47 | 49,28 | 72,97 | 6 | 70 |

- **Registros descartados pelo embargo neste corte:** **`0`** (zero), mantendo isolamento temporal completo entre conclusão de entregas do treino, validação e teste, com boa representatividade de mercados (`6/6`), categorias (`70–74/74`) e distribuição consistente da variável-alvo.

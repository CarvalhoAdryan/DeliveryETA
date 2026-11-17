# DoorDash Delivery ETA — Auditoria Final e Congelamento do Modelo (Etapa 06.2)

> **Escopo e Isolamento:** Esta auditoria utiliza exclusivamente os conjuntos de treino (`138.572`) e validação (`28.808`) definidos na Etapa 05. **O conjunto de teste (`30.041`) permanece 100% isolado e não foi acessado.**

---

## 1. Auditoria do `HistGradientBoostingRegressor` e `early_stopping`

### 1.1 Parâmetros Efetivos Inspecionados (Etapa 06.1)

| Parâmetro | Valor Efetivo (`early_stopping='auto'`) | Observação |
| :--- | :--- | :--- |
| `loss` | `'absolute_error'` | Otimiza L1 (robusto a outliers `> 24h`) |
| `learning_rate` | `0.08` | — |
| `max_iter` | `200` | Limite máximo configurado |
| `max_leaf_nodes` | `63` | — |
| `min_samples_leaf` | `30` | — |
| `l2_regularization` | `0.0` | Padrão do estimador |
| `early_stopping` | `'auto'` | **Ativado automaticamente** porque `n_samples > 10.000` |
| `validation_fraction` | `0.1` | Separa `10%` internos de forma **aleatória** |
| `n_iter_` | `156` | Iterações executadas antes da parada |
| `random_state` | `42` | — |

### 1.2 Investigação Metodológica do `early_stopping`

Como `early_stopping='auto'` utiliza uma divisão **aleatória** de `10%` dentro do conjunto de treino (misturando pedidos futuros e passados para decidir a parada), comparamos três configurações sem utilizar a validação externa ou o teste para ajuste:

| Configuração | `early_stopping` | `n_iter_` | Critério de Parada | Treino (s) | MAE (Val) | RMSE (Val) | MedAE (Val) | P90_AE (Val) | R² (Val) |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atual (`auto`) | `'auto'` | 156 | Split interno **aleatório** 90/10 | 12,22 | 12,1695 | 18,0551 | 9,0774 | 25,7305 | +0,0338 |
| Sem parada | `False` | 200 | Fixo em `max_iter=200` | 11,99 | 12,2297 | 18,1442 | **9,0475** | 26,0011 | +0,0242 |
| **Temporal interno (90/10)** | **`False`** | **20** | **Split interno temporal 90/10 (`created_at`)** | **3,76** | **12,1142** | **17,9219** | 9,2317 | **24,7196** | **+0,0480** |

- **Achado:** Na validação interna estritamente temporal (primeiros `90%` do treino ordenados por `created_at` vs últimos `10%` do treino), o MAE interno atingiu o mínimo em **`20` iterações** (`13,7619`) e degradou monotonicamente até `200` iterações (`15,9140`), indicando sobreajuste a padrões locais da janela de treino com `learning_rate=0.08` e `max_leaf_nodes=63`.
- **Decisão:** Adotar **`early_stopping=False`** com **`max_iter=20`** (definido exclusivamente pelo split interno temporal do treino), eliminando o sorteio aleatório interno e melhorando MAE, RMSE, P90_AE e R².

---

## 2. Auditoria da `RandomForestRegressor`

Os dois experimentos com `RandomForestRegressor` da Etapa 06.1 **não são diretamente equivalentes** devido à estratégia de amostragem e número de árvores:

| Parâmetro | `RF (squared_error)` | `RF (absolute_error)` | Impacto |
| :--- | :--- | :--- | :--- |
| `criterion` | `'squared_error'` | `'absolute_error'` | L1 elimina previsões `> 4.000 min` |
| `n_estimators` | `100` | `50` | Metade das árvores na L1 |
| `max_samples` | `None` (`138.572` por árvore) | `25.000` (`~18%` por árvore) | Subamostragem necessária devido ao custo $O(N^2)$ da L1 |
| `max_depth` / `min_samples_leaf` | `15` / `10` | `15` / `10` | Iguais |
| `random_state` | `42` | `42` | Iguais |
| Tempo de treino | `61,71 s` | `71,87 s` | L1 ainda é mais lenta mesmo com `18%` das amostras e `50` árvores |

---

## 3. Auditoria Completa das Features (`df_feat` = 42 colunas)

- **Catálogo confirmado (Etapa 05):** `22` variáveis em `CONFIRMED_FEATURES`.
- **Utilizadas no modelo conservador:** `21` variáveis em `CONFIRMED_MODEL_FEATURES`.
- **Exclusão explícita:** Apenas `store_id` foi excluído (decisão documentada para evitar explosão dimensional de `6.743` colunas no One-Hot Encoding). Zero exclusões acidentais, zero colunas proibidas e zero variáveis experimentais misturadas.

| Nome da Feature | Classificação | Utilizada no Modelo | Justificativa |
| :--- | :--- | :--- | :--- |
| `market_id`, `store_primary_category`, `order_protocol` | Confirmada | **Sim** | Metadados estáticos do checkout (One-Hot Encoding) |
| `store_id` | Confirmada | **Não** | Excluída do OHE por alta cardinalidade (`6.743` níveis) |
| `total_items`, `subtotal`, `num_distinct_items`, `min_item_price`, `max_item_price` | Confirmada | **Sim** | Composição e valores do carrinho no checkout |
| `estimated_order_place_duration` | Confirmada | **Sim** | `99,9%` constante ligada ao `order_protocol` |
| `created_hour`, `created_dayofweek`, `created_month`, `created_is_weekend`, `created_time_block` | Confirmada | **Sim** | Derivadas do timestamp `created_at` registrado |
| `average_item_price`, `item_price_range`, `invalid_item_price_range`, `distinct_item_ratio`, `is_single_item`, `is_large_order` | Confirmada | **Sim** | Derivadas determinísticas do carrinho no checkout |
| `historical_store_order_count` | Confirmada | **Sim** | Contagem acumulada estritamente anterior por `(store_id, created_at)` |
| `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders` | Experimental | **Não** | Disponibilidade incerta em `created_at` (`95,47%` ausente no mercado 6) |
| `dasher_count_difference`, `busy_greater_than_onshift`, `is_operational_missing`, `outstanding_orders_per_onshift_dasher`, `busy_dashers_ratio` | Experimental | **Não** | Derivadas dos indicadores operacionais de disponibilidade incerta |
| `estimated_store_to_consumer_driving_duration`, `estimated_total_duration` | Experimental | **Não** | Estimativa de deslocamento sem comprovação de congelamento em `created_at` |
| `actual_delivery_time`, `delivery_duration_seconds`, `delivery_duration_minutes`, flags do alvo | Proibida (Alvo) | **Não** | Vazamento direto do alvo |
| `created_at` | Referência | **Não** | Chave temporal usada na divisão e contagem histórica |

---

## 4. Auditoria de Disponibilidade Temporal das Variáveis Experimentais

| Variável | Disponibilidade | Evidência e Risco |
| :--- | :--- | :--- |
| `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders` | **Disponibilidade incerta** | Sem documentação garantindo snapshot exato em `created_at`; `95,47%` ausente no `market_id = 6`. Mantidas como experimentais. |
| `outstanding_orders_per_onshift_dasher`, `busy_dashers_ratio`, `dasher_count_difference`, `busy_greater_than_onshift` | **Disponibilidade incerta** | Derivadas dos contadores acima. Mantidas como experimentais. |
| `estimated_store_to_consumer_driving_duration` | **Disponibilidade incerta** | Não há confirmação se é calculada no checkout ou após alocação do entregador. Mantida como experimental. |

*(Nenhuma delas tem evidência suficiente para ser promovida ao modelo conservador de produção.)*

---

## 5. Configuração Final Congelada (Pronta para Etapa 07)

- **Modelo:** `HistGradientBoostingRegressor`
- **Parâmetros congelados:** `loss="absolute_error"`, `learning_rate=0.08`, `max_iter=20`, `max_leaf_nodes=63`, `min_samples_leaf=30`, `l2_regularization=0.0`, `early_stopping=False`, `random_state=42`
- **Features:** `CONFIRMED_MODEL_FEATURES` (`21` variáveis confirmadas; `store_id` excluído)
- **Target:** `delivery_duration_minutes`
- **Divisão temporal (Etapa 05):** Treino (`138.572`), Validação (`28.808`), Teste (`30.041`, isolado)
- **Pré-processamento:** `ColumnTransformer` ajustado exclusivamente no treino (`SimpleImputer(strategy="median")` nas numéricas; `SimpleImputer(strategy="constant", fill_value="missing")` + `OneHotEncoder(handle_unknown="ignore")` nas categóricas)
- **Versões do ambiente:** Python `3.12.3` | `pandas 2.3.1` | `numpy 2.3.2` | `scikit-learn 1.8.0` | `scipy 1.17.1` | `joblib 1.5.3` | `matplotlib 3.10.6` | `seaborn 0.13.2`

### Justificativa e Pendências para Avaliação Final
- **Justificativa:** Única configuração conservadora que supera o baseline em MAE (`12,11` vs `12,82 min`), RMSE (`17,92` vs `18,68 min`), P90_AE (`24,72` vs `25,67 min`) e R² (`+0,0480` vs `-0,0341`), sem remover outliers do treino e sem depender de divisão aleatória interna no `early_stopping`.
- **Pendências documentadas:** (1) Uso de `store_id` via *target encoding* temporal causal fica para iteração futura; (2) Confirmar com engenharia/produto a disponibilidade em tempo real das razões operacionais e `driving_duration` antes de eventual promoção.
- **Próximo passo:** A configuração está congelada e pronta para a **Etapa 07 (Avaliação Final no Conjunto de Teste)**.

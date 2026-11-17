"""
Módulo de Validação Temporal para o projeto DoorDash Delivery ETA.

Implementa estratégias de divisão temporal (simples e com embargo baseado na
conclusão da entrega `actual_delivery_time`) para evitar vazamento temporal
entre os conjuntos de treino, validação e teste.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


DEFAULT_VAL_START = "2015-02-10 10:00:00"
DEFAULT_TEST_START = "2015-02-14 10:00:00"


@dataclass
class TemporalSplitResult:
    """Contém os subconjuntos da divisão temporal e metadados de embargo."""

    strategy_name: str
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    dropped_train_to_val: pd.DataFrame
    dropped_val_to_test: pd.DataFrame


def temporal_split_simple(
    df: pd.DataFrame,
    val_start: str | pd.Timestamp = DEFAULT_VAL_START,
    test_start: str | pd.Timestamp = DEFAULT_TEST_START,
) -> TemporalSplitResult:
    """
    Divisão temporal simples baseada exclusivamente em `created_at`.

    Não remove pedidos que atravessam os limites de corte; apenas os identifica
    para fins de auditoria e comparação.
    """
    v_ts = pd.Timestamp(val_start)
    t_ts = pd.Timestamp(test_start)
    if v_ts >= t_ts:
        raise ValueError("val_start deve ser estritamente anterior a test_start.")

    train_mask = df["created_at"] < v_ts
    val_mask = (df["created_at"] >= v_ts) & (df["created_at"] < t_ts)
    test_mask = df["created_at"] >= t_ts

    cross_tr_va = df.loc[train_mask & (df["actual_delivery_time"] >= v_ts)].copy()
    cross_va_te = df.loc[val_mask & (df["actual_delivery_time"] >= t_ts)].copy()

    return TemporalSplitResult(
        strategy_name="simple_created_at",
        val_start=v_ts,
        test_start=t_ts,
        train=df.loc[train_mask].copy(),
        val=df.loc[val_mask].copy(),
        test=df.loc[test_mask].copy(),
        dropped_train_to_val=cross_tr_va,
        dropped_val_to_test=cross_va_te,
    )


def temporal_split_with_embargo(
    df: pd.DataFrame,
    val_start: str | pd.Timestamp = DEFAULT_VAL_START,
    test_start: str | pd.Timestamp = DEFAULT_TEST_START,
) -> TemporalSplitResult:
    """
    Divisão temporal com embargo baseado em `actual_delivery_time`.

    Garante que:
    - Pedidos de treino foram criados E entregues antes de `val_start`.
    - Pedidos de validação foram criados em `[val_start, test_start)` E entregues
      antes de `test_start`.
    - Pedidos de teste foram criados a partir de `test_start`.
    Registros que atravessam os cortes são descartados explicitamente.
    """
    v_ts = pd.Timestamp(val_start)
    t_ts = pd.Timestamp(test_start)
    if v_ts >= t_ts:
        raise ValueError("val_start deve ser estritamente anterior a test_start.")

    in_train_window = df["created_at"] < v_ts
    in_val_window = (df["created_at"] >= v_ts) & (df["created_at"] < t_ts)
    in_test_window = df["created_at"] >= t_ts

    cross_tr_va_mask = in_train_window & (df["actual_delivery_time"] >= v_ts)
    cross_va_te_mask = in_val_window & (df["actual_delivery_time"] >= t_ts)

    train_df = df.loc[in_train_window & ~cross_tr_va_mask].copy()
    val_df = df.loc[in_val_window & ~cross_va_te_mask].copy()
    test_df = df.loc[in_test_window].copy()

    return TemporalSplitResult(
        strategy_name="embargo_actual_delivery_time",
        val_start=v_ts,
        test_start=t_ts,
        train=train_df,
        val=val_df,
        test=test_df,
        dropped_train_to_val=df.loc[cross_tr_va_mask].copy(),
        dropped_val_to_test=df.loc[cross_va_te_mask].copy(),
    )


def summarize_split(split: TemporalSplitResult) -> pd.DataFrame:
    """Gera tabela resumo com contagens, janelas temporais, alvo e cobertura."""
    rows = []
    for name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        dur = part["delivery_duration_minutes"]
        rows.append({
            "conjunto": name,
            "n": len(part),
            "created_min": part["created_at"].min(),
            "created_max": part["created_at"].max(),
            "delivery_min": part["actual_delivery_time"].min(),
            "delivery_max": part["actual_delivery_time"].max(),
            "target_p50_min": round(float(dur.median()), 2),
            "target_mean_min": round(float(dur.mean()), 2),
            "target_p90_min": round(float(dur.quantile(0.90)), 2),
            "n_markets": int(part["market_id"].nunique(dropna=True)),
            "n_categories": int(part["store_primary_category"].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)

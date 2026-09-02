"""Заглушка long_term_memory: узлы отдают ожидаемые поля стейта,
а _memory_system_block пропускает в промпт только содержательную память.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.environ.setdefault("GIGACHAT_CREDENTIALS", "test-dummy")

from langgraph_executor.aegra_agents.analytic_orchestrator_v4.nodes import (
    _memory_system_block,
)
from langgraph_executor.aegra_agents.long_term_memory.nodes import (
    make_load_memory_node,
    make_save_memory_node,
)


def test_load_memory_stub_returns_empty_memory():
    node = make_load_memory_node(object())
    out = asyncio.run(node({}, {}))
    assert "отсутствует" in out["memory_context"].lower()
    assert out["memory_loaded"] is True
    assert out["memory_error"] is None


def test_save_memory_stub_mirrors_success():
    node = make_save_memory_node()
    out = asyncio.run(node({}, {}))
    assert out["memory_saved"] is True
    assert out["memory_save_error"] is None


def test_memory_block_filters_empty_memory():
    # Контекст заглушки («…отсутствует») в промпт не попадает.
    assert _memory_system_block({"memory_context": "Долгосрочная память отсутствует."}) is None
    assert _memory_system_block({"memory_context": None}) is None
    assert _memory_system_block({}) is None


def test_memory_block_passes_real_memory():
    block = _memory_system_block({"memory_context": "Руководитель просил сводки кратко."})
    assert block is not None
    assert block.startswith("=== Контекст предыдущих диалогов (долгосрочная память) ===")
    assert "Руководитель просил сводки кратко." in block

"""Единая entry-точка для aegra и PyInstaller.

Загружается aegra по file-path из `aegra.json` (`./link.py:<var>`) и
одновременно служит «якорем» для анализа импортов PyInstaller —
все три подграфа упоминаются здесь, иначе анализатор может не
включить подпакет `aegra_agents` в бандл.
"""
from langgraph_executor.aegra_agents.analytic_orchestrator.graph import (
    graph as analytic_orchestrator_graph,
)
from langgraph_executor.aegra_agents.easyrag.graph import (
    graph as easyrag_graph,
)
from langgraph_executor.aegra_agents.json_analyzer.graph import (
    graph as json_analyzer_graph,
)
from langgraph_executor.aegra_agents.knowledge_agent.graph import (
    graph as knowledge_agent_graph,
)

__all__ = [
    "analytic_orchestrator_graph",
    "easyrag_graph",
    "json_analyzer_graph",
    "knowledge_agent_graph",
]

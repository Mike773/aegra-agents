"""Единая entry-точка для aegra и PyInstaller.

Загружается aegra по file-path из `aegra.json` (`./link.py:<var>`) и
одновременно служит «якорем» для анализа импортов PyInstaller —
все три подграфа упоминаются здесь, иначе анализатор может не
включить подпакет `aegra_agents` в бандл.
"""

from langgraph_executor.aegra_agents.analyst_agent.graph import (
    graph as analyst_agent_graph,
)
from langgraph_executor.aegra_agents.analytic_orchestrator_v4.graph import (
    graph as analytic_orchestrator_v4_graph,
)
from langgraph_executor.aegra_agents.doc_manager.graph import (
    graph as doc_manager_graph,
)
from langgraph_executor.aegra_agents.easyrag.graph import (
    graph as easyrag_graph,
)
from langgraph_executor.aegra_agents.external_doc_loader.graph import (
    graph as external_doc_loader_graph,
)
from langgraph_executor.aegra_agents.gap_resolver.graph import (
    graph as gap_resolver_graph,
)
from langgraph_executor.aegra_agents.json_analyzer_v5.graph import (
    graph as json_analyzer_v5_graph,
)
from langgraph_executor.aegra_agents.metric_enricher.graph import (
    graph as metric_enricher_graph,
)
from langgraph_executor.aegra_agents.kb_chat.graph import (
    graph as kb_chat_graph,
)
from langgraph_executor.aegra_agents.wiki_ingest.graph import (
    graph as wiki_ingest_graph,
)

__all__ = [
    "analyst_agent_graph",
    "analytic_orchestrator_v4_graph",
    "doc_manager_graph",
    "easyrag_graph",
    "external_doc_loader_graph",
    "gap_resolver_graph",
    "json_analyzer_v5_graph",
    "kb_chat_graph",
    "metric_enricher_graph",
    "wiki_ingest_graph",
]

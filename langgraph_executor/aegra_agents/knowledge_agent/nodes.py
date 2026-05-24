from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat import GigaChat

from .prompts import SYSTEM_PROMPT
from .state import KnowledgeState


def make_retrieve_node():
    def retrieve(state: KnowledgeState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable", {})
        top_k = int(cfg.get("top_k", 3))
        collection = cfg.get("knowledge_collection", "default")

        query = state.get("query") or _last_user_text(state)

        # TODO: реальный pgvector / FTS поиск.
        # Пример будущей логики:
        #   from ..shared.clients import create_gigachat_client
        #   embedding = create_gigachat_client().create_embedding(query)
        #   snippets = pgvector_search(collection, embedding, k=top_k)
        snippets = [
            {"text": f"[stub fragment about: {query!r}]", "source": f"{collection}#stub-{i}"}
            for i in range(top_k)
        ]
        return {"query": query, "snippets": snippets}

    return retrieve


def make_generate_node(llm: GigaChat):
    def generate(state: KnowledgeState, config: RunnableConfig) -> dict:
        cfg = (config or {}).get("configurable", {})
        system_prompt = cfg.get("system_prompt_override") or SYSTEM_PROMPT

        snippets = state.get("snippets") or []
        snippets_text = "\n\n".join(
            f"[{s.get('source','?')}] {s.get('text','')}" for s in snippets
        ) or "(нет фрагментов)"
        user_text = (
            f"Вопрос: {state.get('query','')}\n\nФрагменты:\n{snippets_text}"
        )

        ai = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ])
        return {"messages": [ai], "answer": ai.content}

    return generate


def _last_user_text(state: KnowledgeState) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""

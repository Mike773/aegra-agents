"""Заглушка клиента внешнего источника документов.

Единственный метод — ``fetch_documents()`` — возвращает массив JSON-объектов
вида ``{"direction_id": <ключ направления>, "text": <текст документа>}``.
Когда появится реальная интеграция (HTTP/очередь), меняется только тело
метода — формат ответа и точка подключения (граф ``external_doc_loader``)
остаются прежними.

``direction_id`` внешнего источника соответствует ``direction_key`` в схеме
``wiki_rag`` — маппинг 1:1, без преобразований.
"""
from __future__ import annotations

from typing import Any

# Демо-данные, чтобы граф можно было прогнать end-to-end без реального
# источника. При подключении реальной интеграции — удалить.
_DEMO_DOCUMENTS: list[dict[str, Any]] = [
    {
        "direction_id": "demo",
        "text": (
            "Колобок — хлебобулочное изделие сферической формы. "
            "Испечён из остатков муки, самостоятельно покинул место производства."
        ),
    },
    {
        "direction_id": "demo",
        "text": (
            "Лиса — хищник семейства псовых. Известна успешным перехватом "
            "хлебобулочных изделий на маршруте их следования."
        ),
    },
]


def transform_documents(raw_items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Преобразовать сырой массив внешнего источника в документы загрузчика.

    Каждый элемент входного массива — отдельный документ. Из него берём:

    * текст — ``dataset.result.response.message.sources[0].answer``
      (``sources`` у message всегда один);
    * направление — первый тег ``metadata.tags`` этого же source, из тега
      берётся часть после префикса: ``kd_35`` → ``35``.

    Возвращает массив ``{"direction_id": str, "text": str}`` — формат
    ``fetch_documents``. Элементы без текста или без валидного тега молча
    пропускаются: их всё равно нечем грузить.
    """
    documents: list[dict[str, Any]] = []
    for item in raw_items or []:
        try:
            source = item["dataset"]["result"]["response"]["message"]["sources"][0]
        except (KeyError, IndexError, TypeError):
            continue
        text = str(source.get("answer") or "").strip()
        metadata = source.get("metadata") or {}
        tags = metadata.get("tags") or []
        tag = str(tags[0]).strip() if tags else ""
        _, _, direction = tag.partition("_")
        direction = direction.strip()
        if not text or not direction:
            continue
        documents.append({"direction_id": direction, "text": text})
    return documents


class ExternalDocumentSource:
    """Клиент внешнего источника документов (пока заглушка)."""

    async def fetch_documents(self) -> list[dict[str, Any]]:
        """Все доступные документы источника.

        Элемент: ``{"direction_id": str, "text": str}``. Заглушка отдаёт
        статичные демо-данные; реальная реализация будет ходить во внешний API.
        """
        return list(_DEMO_DOCUMENTS)


__all__ = ["ExternalDocumentSource", "transform_documents"]

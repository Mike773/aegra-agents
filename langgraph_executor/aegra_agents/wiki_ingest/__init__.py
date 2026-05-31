"""Подагент `wiki_ingest`: загрузка документов в wiki.

Порт ingest-стороны /Users/mikhailorlov/Development/easyRag под aegra. На вход —
``direction_key``; агент читает необработанные документы из ``wiki_rag.source_doc``
(``processed_at IS NULL``) и прогоняет каждый через полный пайплайн:
chunk → domain brief → LLM-extract сущностей → LLM-резолвер (судья + merge) в
``wiki_page``/``wiki_section`` → backlinks. ORM-модели и сессия БД — общие с
подагентом ``easyrag`` (схема ``wiki_rag``, подключение из aegra).
"""

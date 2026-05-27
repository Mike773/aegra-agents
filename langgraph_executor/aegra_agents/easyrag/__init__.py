"""Подагент `easyrag`: RAG-выборка по wiki + запись gap'а.

Узкий порт из /Users/mikhailorlov/Development/easyRag. Отличия:
- хранение в схеме ``wiki_rag`` (БД — та же, что у aegra);
- разделение по ``direction_key`` (фильтр выборки и поле gap'а).
"""

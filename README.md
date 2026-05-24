# aegra-agents

Шаблоны трёх независимых LangGraph-агентов для self-hosted сервиса
[aegra](https://github.com/ibbybuilds/aegra) (open-source альтернатива LangGraph Platform).
LLM и эмбеддинги — через `langchain-gigachat`.

## Состав

| Граф | Назначение |
|---|---|
| `knowledge_agent` | Ответ по корпоративной базе знаний (retrieve + generate). Источник — пока заглушка под pgvector / FTS. |
| `json_analyzer` | Разбор переданного JSON-документа (parse + LLM-анализ). |
| `analytic_orchestrator` | Диалоговый HITL-граф: `interrupt` → роутер → вызов нужного подграфа → финальный ответ → новая итерация. |

Все три графа зарегистрированы в `aegra.json` и доступны через стандартный
API aegra (`/assistants`, `/threads`, `/runs/stream`) независимо.
`analytic_orchestrator` внутри использует `knowledge_agent` и `json_analyzer`
как подграфы (через явный `subgraph.invoke(...)` в обёрточном узле).

## Раскладка

```
.
├── aegra.json                                              # регистрация графов и http-приложения
└── langgraph_executor/
    ├── agent/services/clients/gigachat.py                  # заглушка GigaChatClient (заменить на реальный)
    └── aegra_agents/
        ├── shared/clients.py                               # тонкая обёртка над GigaChatClient
        ├── knowledge_agent/{state,prompts,nodes,graph}.py
        ├── json_analyzer/{state,prompts,nodes,graph}.py
        └── analytic_orchestrator/{state,prompts,nodes,graph}.py
```

## Единый контракт каждого графа

```python
# graph.py
def build_graph(llm: GigaChat):
    g = StateGraph(MyState)
    g.add_node(...)
    ...
    return g.compile()                  # checkpointer подставит aegra

llm = create_gigachat_client().get_llm()
graph = build_graph(llm)                # имя `graph` ищет aegra.json
```

Параметры приходят через `RunnableConfig.configurable` и читаются внутри узлов:

```python
def my_node(state, config: RunnableConfig):
    cfg = (config or {}).get("configurable", {})
    top_k = cfg.get("top_k", 3)
```

## Configurable по графам

| Граф | Ключи |
|---|---|
| `knowledge_agent` | `knowledge_collection`, `top_k`, `system_prompt_override` |
| `json_analyzer` | `schema_hint`, `max_depth`, `system_prompt_override` |
| `analytic_orchestrator` | `enabled_subagents` (по умолчанию `["knowledge","json"]`), `language`, `system_prompt_override` |

## GigaChat-клиент

`langgraph_executor/agent/services/clients/gigachat.py` — **заглушка** с публичным API:

```python
class GigaChatClient:
    def get_llm(self) -> GigaChat: ...
    def create_embedding(self, text: str) -> list[float]: ...

def create_gigachat_client() -> GigaChatClient: ...
```

При интеграции в реальный сервис этот файл вытесняется настоящей реализацией
с теми же сигнатурами.

ENV для запуска заглушки:

```
GIGACHAT_CREDENTIALS=...
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_BASE_URL=                  # пусто = дефолт
GIGACHAT_VERIFY_SSL=false
```

## Локальная проверка

После установки зависимостей (`pip install -e .` или `pip install langgraph langchain-gigachat`):

```bash
python -c "from langgraph_executor.aegra_agents.knowledge_agent.graph import graph; \
           print(graph.get_graph().draw_ascii())"
python -c "from langgraph_executor.aegra_agents.json_analyzer.graph import graph; \
           print(graph.get_graph().draw_ascii())"
python -c "from langgraph_executor.aegra_agents.analytic_orchestrator.graph import graph; \
           print(graph.get_graph().draw_ascii())"
```

## Запуск в aegra

`aegra.json` уже настроен. Поднимите сервис aegra стандартным способом
(docker-compose / uvicorn `langgraph_executor.plugins.app:app` — `plugins/app.py`
не входит в этот репозиторий и должен быть в вашем рабочем проекте).

Пример вызова `analytic_orchestrator` с HITL через
[`langgraph-sdk`](https://pypi.org/project/langgraph-sdk/):

```python
from langgraph_sdk import get_client
from langgraph_sdk.schema import Command

client = get_client(url="http://localhost:8000")
thread = await client.threads.create()

async for ev in client.runs.stream(
    thread["thread_id"], "analytic_orchestrator",
    input={"messages": []},
    config={"configurable": {"enabled_subagents": ["knowledge", "json"]}},
):
    print(ev)

# возобновление после interrupt
async for ev in client.runs.stream(
    thread["thread_id"], "analytic_orchestrator",
    command=Command(resume="Расскажи про продукт X"),
):
    print(ev)
```

## TODO

- Реальный retrieval в `knowledge_agent.retrieve` (pgvector / FTS) через `create_embedding`.
- Реальный анализ JSON и структурированный output в `json_analyzer.analyze`.
- Декларация `config_schema` через pydantic-модели вместо чтения словарём.
- Тесты.

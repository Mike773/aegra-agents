import json
import datetime
import httpx
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.parse import urlencode

import argparse
import json
from pathlib import Path
from typing import Any
from settings import settings

DEFAULT_REQUEST_TIMEOUT = 60
AGENT_DATASET_ENDPOINT = "/api/v2/dataset/"

def _pick(detail: dict[str, Any], parent: dict[str, Any], key: str) -> Any:
    """Берёт key из detail, если задано (не None), иначе из parent."""
    if key in detail and detail[key] is not None:
        return detail[key]
    return parent.get(key)


def _history_clones(base: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Возвращает клоны base с подменёнными dt/fact/plan/bs/calc_period из history."""
    clones: list[dict[str, Any]] = []
    for h in history or []:
        clone = dict(base)
        clone["date"] = h.get("dt", base.get("date"))
        clone["calc_period"] = h.get("calc_period", base.get("calc_period"))
        clone["fact"] = h.get("fact")
        clone["plan"] = h.get("plan")
        clone["star_received"] = h.get("star_received")
        clone["is_star_metric"] = h.get("is_star_metric")
        clone["benchmark"] = h.get("bs")
        clone["child_metrics"] = []
        clones.append(clone)
    return clones


def convert_metric(metric: dict[str, Any]) -> list[dict[str, Any]]:
    """Разворачивает одну raw-метрику в плоский список loader-узлов.

    Возвращает список, который родитель кладёт в child_metrics своего агрегата
    (или, для топ-уровня, в metrics).
    """
    out: list[dict[str, Any]] = []

    converted_children: list[dict[str, Any]] = []
    for child in metric.get("children_metrics") or []:
        converted_children.extend(convert_metric(child))

    has_fact = metric.get("fact") is not None or metric.get("star_received") is not None

    if has_fact:
        aggregate = {
            "id": metric.get("metric_id"),
            "metric_name": metric.get("metric_name"),
            "metric_description": metric.get("metric_description"),
            "metric_type": metric.get("metric_type"),
            "measure_type": metric.get("measure_type"),
            "date": metric.get("dt"),
            "calc_period": metric.get("calc_period"),
            "fact": metric.get("fact"),
            "star_received": metric.get("star_received"),
            "is_star_metric": metric.get("is_star_metric"),
            "plan": metric.get("plan"),
            "benchmark": metric.get("bs"),
            "element": None,
            "child_metrics": converted_children,
        }
        if metric.get("influence_percent") is not None:
            aggregate["influent_percent"] = metric.get("influence_percent")
        out.append(aggregate)
        out.extend(_history_clones(aggregate, metric.get("history") or []))
    else:
        # Агрегат не эмитим; чтобы не потерять детей, поднимаем их на тот же
        # уровень, что и детали (вариант B из плана).
        out.extend(converted_children)

    for detail in metric.get("details") or []:
        node = {
            "id": detail.get("metric_id") or metric.get("metric_id"),
            "metric_name": metric.get("metric_name"),
            "metric_description": metric.get("metric_description"),
            "metric_type": _pick(detail, metric, "metric_type"),
            "measure_type": _pick(detail, metric, "measure_type"),
            "date": detail.get("dt"),# if detail.get("dt") is not None else metric.get("dt"),
            "calc_period": _pick(detail, metric, "calc_period"),
            "fact": detail.get("fact"),
            "star_received": detail.get("star_received"),
            "is_star_metric": detail.get("is_star_metric"),
            "plan": detail.get("plan"), #if detail.get("plan") is not None else metric.get("plan"),
            "benchmark": detail.get("bs"),# if detail.get("bs") is not None else metric.get("bs"),
            "element": detail.get("element_name"),
            "child_metrics": [],
        }
        infl = detail.get("influence_percent")
        if infl is None:
            infl = metric.get("influence_percent")
        if infl is not None:
            node["influent_percent"] = infl
        out.append(node)
        out.extend(_history_clones(node, detail.get("history") or []))

    return out


def convert_dataset(raw: dict[str, Any]) -> dict[str, Any]:
    """Оборачивает результат convert_metric в envelope {me, employees}."""
    metrics_out: list[dict[str, Any]] = []
    for m in raw.get("metrics") or []:
        metrics_out.extend(convert_metric(m))
    return metrics_out


def collect_aggregates_ids(metrics: list[dict[str, Any]]) -> list[str]:
    """Уникальные aggregates_ids по дереву сырых метрик (children_metrics),
    порядок первого появления. convert_metric это поле отбрасывает, поэтому
    собираем его ДО конвертации и отдаём списком на уровне персоны."""
    out: list[str] = []
    seen: set[str] = set()

    def _walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            for agg_id in node.get("aggregates_ids") or []:
                if agg_id is None:
                    continue
                text = str(agg_id).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append(text)
            _walk(node.get("children_metrics") or [])

    _walk(metrics)
    return out

class GetBatchAgentAggregateDatasetByFiltersComponent:
    """Запрос batch-агрегатов peer-групп по уровням SBER/TB/GOSB.

    Args: как у ``GetBatchAgentDatasetByFiltersComponent`` — те же
    ``dataset_name`` и ``filters``.

    Ответ: список ``{"dataset": {"level", "metrics": [...]}}``, у метрики —
    ``aggregates`` (текущий срез + ``history``) и дети в ``children_metrics``.
    """
    url = settings.agent_dataset_url
    verify_ssl = False
    timeout = DEFAULT_REQUEST_TIMEOUT
    metric_group_name = "metrics_for_agent_analyst_test_agg"

    def __init__(
        self,
        dataset_name: str,
        filters: dict[str, Any] | str,
    ) -> None:
        print("START AGGGGGGGGGRREGATE")
        self.dataset_name = dataset_name
        self.filters = filters
    
    def _fetch_records(self, client: httpx.Client) -> List[Dict]:
        """Выполняет запрос к API и возвращает список записей."""
        result = []

        params = {
            "project_name": self.dataset_name,
            "add_filters": json.dumps(self.filters, ensure_ascii=False)
        }

        url = f"{self.url.rstrip('/')}{AGENT_DATASET_ENDPOINT}?{urlencode(params)}"

        try:
            response = client.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                result.extend(data)
            elif isinstance(data, dict):
                result.extend(data.get("records", [data] if data else []))
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise Exception(f"Ошибка запроса: {str(e)}")

        return result
    
    def fetch_documents(self) -> list[dict[str, Any]]:
        """Все доступные документы источника.

        Элемент: ``{"direction_id": str, "text": str}``. Заглушка отдаёт
        статичные демо-данные; реальная реализация будет ходить во внешний API.
        """
        response = []
        with httpx.Client(
            verify=self.verify_ssl,
            timeout=DEFAULT_REQUEST_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        ) as client:
            response=self._fetch_records(client)
        # print(response)
        return response

    def build_json_output(self) -> list[dict[str, Any]]:   
        print("START")     
        return self.fetch_documents()

class GetBatchAgentDatasetByFiltersComponent:
    display_name = "Get Batch Agent Dataset by Filters"
    description = "Получение актуальных датасетов по идентификаторам из addFiltersObjects (с учётом load_id)."
    icon = "Isu"
    name = "GetBatchAgentDatasetByFilters"

    url = settings.agent_dataset_url
    verify_ssl = False
    timeout = DEFAULT_REQUEST_TIMEOUT

    metric_group_name = None
    input_json = None

    FILTER_PRIORITIES = {
    "USER": 1,
    "POSITION": 2,
    "ORG_STRUCT_ELEMENT": 3,
    }
    

    def __init__(self,dataset_name:str,filters:str):
        self.input_json=filters
        self.metric_group_name=dataset_name
        print(filters)

        

    def _get_filter_priority(self, object_type: str) -> int:
        """Возвращает приоритет типа фильтра."""
        return self.FILTER_PRIORITIES.get(object_type, 99)

    def _extract_person_filters(self, data: Dict) -> Tuple[Dict[str, List[Dict]], List[Dict], List[Dict]]:
        """
        Извлекает фильтры для каждого человека.
        Возвращает: (person_filters, employees_data, me_filters)
        """
        
        person_filters = {}
        
        # Обработка me
        me_data = data.get("me", {})
        me_filters = [
            f for f in me_data.get("addFiltersObjects", [])
            if f.get("object_type") and f.get("object_id") is not None
        ]
        if me_filters:
            person_filters["me"] = me_filters

        # Обработка employees
        employees_data = data.get("employees", [])
        for emp in employees_data:
            person_id = emp.get("posKey") or emp.get("tabNum")
            if not person_id:
                continue
            emp_filters = [
                f for f in emp.get("addFiltersObjects", [])
                if f.get("object_type") and f.get("object_id") is not None
            ]
            if emp_filters:
                person_filters[person_id] = emp_filters

        return person_filters, employees_data, me_filters

    def _filter_unique_filters(self, person_filters: List[Dict], me_filters: List[Dict]) -> List[Dict]:
        """
        Отбрасывает фильтры сотрудника, которые совпадают с фильтрами руководителя.
        Возвращает только уникальные для сотрудника фильтры.
        """
        me_filter_set = {(f["object_type"], str(f["object_id"])) for f in me_filters}
        
        unique_filters = [
            f for f in person_filters
            if (f["object_type"], str(f["object_id"])) not in me_filter_set
        ]
        
        return unique_filters

    def _create_filter_set(self, filters: List[Dict]) -> Set[Tuple[str, str]]:
        """Создаёт множество кортежей (object_type, object_id) из списка фильтров."""
        return {(f["object_type"], str(f["object_id"])) for f in filters}

    def _extract_date_from_load_id(self, load_id: str) -> Optional[datetime.date]:
        """
        Извлекает дату из load_id (ожидается формат ..._YYYYMMDD).
        Возвращает date или None, если не удалось распарсить.
        """
        try:
            parts = load_id.split('_')
            date_str = parts[-1]
            return datetime.datetime.strptime(date_str, "%Y%m%d").date()
        except (ValueError, IndexError):
            return None

    def _get_latest_record(self, records: List[Dict]) -> Optional[Dict]:
        """
        Из списка записей выбирает запись с максимальной датой load_id.
        Если дату не удалось определить ни для одной записи, возвращает первую.
        """
        if not records:
            return None

        best_record = records[0]
        best_date = None
        if "load_id" in best_record:
            best_date = self._extract_date_from_load_id(best_record["load_id"])

        for record in records[1:]:
            if "load_id" not in record:
                continue
            current_date = self._extract_date_from_load_id(record["load_id"])
            if current_date is None:
                continue
            if best_date is None or current_date > best_date:
                best_date = current_date
                best_record = record

        return best_record

    def _find_matching_records(self, records: List[Dict], person_filter_set: Set[Tuple[str, str]]) -> List[Dict]:
        """
        Возвращает все записи, у которых add_filters пересекается с фильтрами человека.
        """
        matching = []
        for record in records:
            record_filters = record.get("add_filters")
            if not record_filters or not isinstance(record_filters, list):
                continue
            record_filter_set = self._create_filter_set(record_filters)
            if person_filter_set & record_filter_set:
                matching.append(record)
        return matching
        
    def _fetch_records(self, client: httpx.Client, filters: List[Dict]) -> List[Dict]:
        """Выполняет запрос к API и возвращает список записей."""
        result = []
    
        for filter in filters:
            request_param = []
            request_param.append(filter)
            params = {
                "project_name": self.metric_group_name,
                "add_filters": json.dumps(request_param, ensure_ascii=False)
            }
            
            url = f"{self.url.rstrip('/')}{AGENT_DATASET_ENDPOINT}?{urlencode(params)}"
            
            try:
                response = client.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    result.extend(data)
                elif isinstance(data, dict):
                    result.extend(data.get("records", [data] if data else []))
            except httpx.HTTPStatusError as e:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                raise Exception(f"Ошибка запроса: {str(e)}")
                
        return result

    def _process_person(self, client: httpx.Client, person_id: str, filters: List[Dict]) -> Optional[Any]:
        """
        Обрабатывает одного человека с учётом приоритета фильтров:
        1. Группирует фильтры по object_type.
        2. Проверяет типы от самых специфичных (USER) к общим (ORG_STRUCT_ELEMENT).
        3. Если находит данные по более специфичному фильтру, игнорирует остальные.
        """
        
        # 1. Группируем фильтры по object_type
        filters_by_type = {}
        for f in filters:
            obj_type = f.get("object_type")
            if obj_type:
                filters_by_type.setdefault(obj_type, []).append(f)
                
        if not filters_by_type:
            return None

        # 2. Сортируем типы фильтров по приоритету (от самого важного к менее важному)
        sorted_types = sorted(filters_by_type.keys(), key=lambda t: self._get_filter_priority(t))
        
        # 3. Итеративно запрашиваем данные, начиная с самых специфичных
        for obj_type in sorted_types:
            current_filters = filters_by_type[obj_type]
            
            try:
                # Запрашиваем записи только для фильтров текущего типа
                records = self._fetch_records(client, current_filters)
                if not records:
                    continue
                
                # Проверяем пересечение ТОЛЬКО по текущему типу фильтров
                current_filter_set = self._create_filter_set(current_filters)
                matching_records = self._find_matching_records(records, current_filter_set)
                
                # Если нашли подходящие записи, выбираем самую свежую и возвращаем её.
                # Цикл прерывается (break), более общие фильтры не проверяются!
                if matching_records:
                    latest_record = self._get_latest_record(matching_records)
                    if latest_record:
                        return latest_record.get("dataset", latest_record)
                        
            except Exception:
                # Если по конкретному типу произошла ошибка API, пробуем следующий тип
                continue
                
        return None

    def build_json_output(self) -> str:
        """Возвращает результаты в формате JSON."""
        
        try:
            input_data = json.loads(self.input_json)
        except json.JSONDecodeError as e:
            self.status = f"Ошибка парсинга JSON: {e}"
            return json.dumps({"me": {}, "employees": []})

        try:
            # Получаем также me_filters для последующей фильтрации общих ключей
            person_filters, employees_data, me_filters = self._extract_person_filters(input_data)
            
            if not person_filters:
                return json.dumps({"me": {}, "employees": []})

            results = {}
            
            with httpx.Client(
                verify=self.verify_ssl,
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            ) as client:
                for person_id, filters in person_filters.items():
                    # Если это не "me", фильтруем общие с руководителем фильтры
                    if person_id != "me" and me_filters:
                        unique_filters = self._filter_unique_filters(filters, me_filters)
                        
                        # Если после фильтрации ничего не осталось, пропускаем сотрудника
                        if not unique_filters:
                            continue
                        
                        # Используем приоритизацию для уникальных фильтров
                        dataset = self._process_person(client, person_id, unique_filters)
                    else:
                        # Для "me" используем все фильтры (приоритизация всё равно работает)
                        dataset = self._process_person(client, person_id, filters)
                        
                    if dataset is not None:
                        results[person_id] = dataset

            # Формируем выходные данные — включаем только тех, по кому нашли данные
            output = {}
            
            # Обработка me
            me_dataset = results.get("me")
            if me_dataset:
                me_data = input_data.get("me", {})
                me_output = {
                    "fio": me_data.get("title"),
                    "depart": me_data.get("depart", {}).get("title") if me_data.get("depart") else None,
                    "post": me_data.get("post", {}).get("title") if me_data.get("post") else None,
                    "position": me_data.get("position", {}).get("title") if me_data.get("position") else None,
                    "segment": me_data.get("segments", [{}])[0].get("title") if me_data.get("segments") else None,
                }

                me_output.update(me_dataset)
                # ids предагрегатов — из СЫРЫХ метрик, до замены конвертированными.
                me_output["aggregates_ids"] = collect_aggregates_ids(
                    me_output.get("metrics") or []
                )
                me_output["metrics"] = convert_dataset(me_output)
                output["me"] = me_output
            
            # Обработка employees
            employees_output = []
            for emp in employees_data:
                person_id = emp.get("posKey") or emp.get("tabNum")
                emp_dataset = results.get(person_id)
                if emp_dataset:
                    emp_output = {
                        "fio": emp.get("userName"),
                        "depart": emp.get("departName"),
                        "post": emp.get("postName"),
                    }
                    emp_output.update(emp_dataset)
                    emp_output["aggregates_ids"] = collect_aggregates_ids(
                        emp_output.get("metrics") or []
                    )
                    emp_output["metrics"] = convert_dataset(emp_output)
                    employees_output.append(emp_output)
            if employees_output:
                output["employees"] = employees_output

            self.status = f"Найдено датасетов: {len(results)}"
            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            self.status = f"Ошибка: {str(e)}"
            return json.dumps({"me": {}, "employees": []})
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kungur Sources Crawler & Text Extractor & People Parser (prototype)

Что делает за один прогон:
1) Обходит https://kungur-sources.github.io/ (внутренние ссылки, глубина по умолчанию 2)
2) Для каждой HTML-страницы:
   - скачивает и извлекает основной текст (чистит меню/навигацию)
   - сохраняет текст в data/kungur-sources/docs/<slug>.txt
   - пишет запись в data/kungur-sources/index.jsonl (source_url, title, path, hash, collected_at)
3) Бегло парсит персоналии (имя/возраст/роль) и сохраняет CSV в data/kungur-sources/tables/<slug>.csv
   - создаёт сводный data/kungur-sources/tables/_all_people.csv

Запуск локально:
  python scripts/fetch_kungur_sources.py

Примечания:
- Эвристики извлечения текста и персоналий упрощены. Подгоняются под реальный контент.
- Соблюдает вежливый rate limit (1 запрос/сек).
"""

import os
import re
import csv
import json
import time
import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

# ---------- Конфиг ----------
START_URL = "https://kungur-sources.github.io/"
DOMAIN = "kungur-sources.github.io"
MAX_PAGES = 2000        # предохранитель
MAX_DEPTH = 2           # глубина обхода
REQUEST_TIMEOUT = 30
RATE_LIMIT_SECONDS = 1

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "kungur-sources")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
TABLES_DIR = os.path.join(DATA_DIR, "tables")
INDEX_PATH = os.path.join(DATA_DIR, "index.jsonl")
ALL_PEOPLE_PATH = os.path.join(TABLES_DIR, "_all_people.csv")

HEADERS = {
    "User-Agent": "GenealogyAggregatorBot/0.1 (+https://github.com/ant69)"
}

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Вспомогательные ----------

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def normalize_space(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def slugify(path: str) -> str:
    """
    Делает компактное имя файла из пути URL.
    """
    # только часть пути, без домена и query
    parsed = urlparse(path)
    p = parsed.path
    if p.endswith("/"):
        p = p[:-1]
    if not p:
        p = "index"
    # Заменяем недопустимые символы
    p = unicodedata.normalize("NFKD", p)
    p = p.encode("ascii", "ignore").decode("ascii")
    p = p.strip("/")
    p = re.sub(r"[^A-Za-z0-9/_\-]+", "-", p)
    p = p.replace("/", "_")
    if not p:
        p = "page"
    return p

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def is_internal_link(href: str) -> bool:
    if not href:
        return False
    href = href.strip()
    if href.startswith("mailto:") or href.startswith("javascript:"):
        return False
    if href.startswith("#"):
        return True
    parsed = urlparse(href)
    if parsed.netloc and parsed.netloc != DOMAIN:
        return False
    # отсекаем очевидные ассеты
    if parsed.path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".pdf", ".zip")):
        return False
    return True

def choose_main_node(soup: BeautifulSoup):
    """
    Эвристика выбора основной колонки текста:
    1) <main> или <article>
    2) самый длинный <div> по числу букв (последний шанс)
    """
    for tag in ("main", "article"):
        node = soup.find(tag)
        if node and len(node.get_text(strip=True)) > 200:
            return node

    # выкидываем навигацию/футеры/сайдбары
    for selector in ["nav", "header", "footer", "aside"]:
        for el in soup.find_all(selector):
            el.decompose()

    # ищем самый «текстовый» div
    best = None
    best_len = 0
    for div in soup.find_all("div"):
        txt = div.get_text(" ", strip=True)
        l = len(txt)
        if l > best_len:
            best_len = l
            best = div
    return best or soup.body or soup

def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    return ""

def fetch(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    # Коррекция кодировки, если не задана
    if not resp.encoding or resp.encoding.lower() in ("ascii", "utf-8"):
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp

# ---------- Индекс ----------

def load_index() -> dict:
    """
    Читаем index.jsonl -> dict[url] = last_record
    """
    index = {}
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    index[rec["source_url"]] = rec
                except Exception:
                    continue
    return index

def append_index(rec: dict):
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- Парсер персоналий (черновой) ----------

ROLE_MARKERS = [
    ("дети", "child"),
    ("сын", "son"),
    ("дочь", "daughter"),
    ("брат", "brother"),
    ("братья", "brother"),
    ("сестра", "sister"),
    ("внук", "grandson"),
    ("внучата", "grandchildren"),
    ("племянник", "nephew"),
    ("племянники", "nephews"),
    ("пасынок", "stepson"),
    ("зять", "son-in-law"),
    ("подворник", "lodger"),
    ("приемыш", "foster"),
]

HEAD_RE = re.compile(r'^([А-ЯЁA-Z][\w\-]+)\s+([А-ЯЁA-Z][\w\-]+)\s+сын\s+([А-ЯЁA-Z][\w\-]+)\s+(\d{1,3})', re.IGNORECASE)
NAME_AGE_RE = re.compile(r'([А-ЯЁA-Z][\w\-]+)\s+(\d{1,3})(?:\b|$)', re.IGNORECASE)
SETTLEMENT_RE = re.compile(r'^(?:села|село|деревни|деревня|острожку|острог|г\.)\s+.*$', re.IGNORECASE)
FOLIO_RE = re.compile(r'/л\.?\s*([\d]+(?:об\.)?(?:\.\d)?)/?', re.IGNORECASE)

def detect_role(segment: str):
    for ru, role in ROLE_MARKERS:
        if ru in segment:
            return role, ru
    return None, None

def parse_people_from_text(doc_text: str):
    """
    Возвращает список словарей: line_no, settlement, folio, role, name, patronymic, family, age, raw_segment.
    """
    rows = []
    lines = [l.strip() for l in doc_text.split("\n")]
    current_settlement = None
    current_folio = None

    for i, raw_line in enumerate(lines, start=1):
        if not raw_line:
            continue

        for m in FOLIO_RE.finditer(raw_line):
            current_folio = m.group(1)

        if SETTLEMENT_RE.match(raw_line):
            current_settlement = raw_line

        # дробим по ;
        segments = [s.strip() for s in raw_line.split(";") if s.strip()]
        if not segments:
            continue

        head_parsed = HEAD_RE.search(segments[0])
        head_name = None
        if head_parsed:
            head_name = head_parsed.group(1)
            rows.append({
                "line_no": i,
                "settlement": current_settlement,
                "folio": current_folio,
                "role": "head",
                "name": head_parsed.group(1),
                "patronymic": head_parsed.group(2),
                "family": head_parsed.group(3),
                "age": head_parsed.group(4),
                "raw_segment": segments[0],
            })
            segments[0] = segments[0][head_parsed.end():].strip()

        for seg in segments:
            role, _ru = detect_role(seg)
            for m in NAME_AGE_RE.finditer(seg):
                nm, age = m.groups()
                if head_name and nm == head_name and role is None:
                    continue
                rows.append({
                    "line_no": i,
                    "settlement": current_settlement,
                    "folio": current_folio,
                    "role": role if role else "other",
                    "name": nm,
                    "patronymic": None,
                    "family": None,
                    "age": age,
                    "raw_segment": seg,
                })

    return rows

# ---------- Основной обход ----------

@dataclass
class QueueItem:
    url: str
    depth: int

def crawl():
    seen = set()
    q = [QueueItem(START_URL, 0)]
    index = load_index()
    saved_pages = 0
    parsed_pages = 0

    all_people_rows = []

    while q and saved_pages < MAX_PAGES:
        item = q.pop(0)
        url, depth = item.url, item.depth
        url = urldefrag(url)[0]  # без #fragment

        if url in seen:
            continue
        seen.add(url)

        try:
            time.sleep(RATE_LIMIT_SECONDS)
            resp = fetch(url)
        except Exception as e:
            print(f"[WARN] fetch failed {url}: {e}")
            continue

        if resp.status_code != 200 or "text/html" not in (resp.headers.get("Content-Type","")):
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        title = extract_title(soup)

        # Извлекаем основной текст
        main = choose_main_node(soup)
        text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
        text = normalize_space(text)

        # Подготовим файл
        slug = slugify(url)
        txt_path = os.path.join(DOCS_DIR, f"{slug}.txt")

        # Мета-заголовок + очищенный текст
        content_hash = sha256_text(text)
        header = [
            "---",
            f"source_url: {url}",
            f"title: \"{title.replace('\"', '\\\"')}\"",
            f"collected_at: \"{now_iso()}\"",
            f"content_hash: \"sha256:{content_hash}\"",
            "lang: ru",
            "---",
            "",
        ]
        out_text = "\n".join(header) + text + "\n"

        # Проверим, есть ли в индексе и не поменялся ли хэш
        prev = index.get(url)
        need_write = True
        if prev and prev.get("content_hash") == content_hash:
            need_write = False

        if need_write:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(out_text)
            saved_pages += 1
            rec = {
                "source_url": url,
                "title": title,
                "path": os.path.relpath(txt_path, REPO_ROOT).replace("\\", "/"),
                "content_hash": content_hash,
                "collected_at": now_iso(),
            }
            append_index(rec)
            index[url] = rec
            print(f"[OK] saved: {url} -> {rec['path']}")

        # Построим таблицу персоналий по этой странице
        people = parse_people_from_text(text)
        if people:
            csv_path = os.path.join(TABLES_DIR, f"{slug}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["source_url", "title", "line_no", "settlement", "folio",
                            "role", "name", "patronymic", "family", "age", "raw_segment"])
                for r in people:
                    w.writerow([url, title, r["line_no"], r["settlement"], r["folio"],
                                r["role"], r["name"], r["patronymic"], r["family"], r["age"], r["raw_segment"]])
            parsed_pages += 1
            all_people_rows.extend(
                [[url, title, r["line_no"], r["settlement"], r["folio"],
                  r["role"], r["name"], r["patronymic"], r["family"], r["age"], r["raw_segment"]]
                 for r in people]
            )
            print(f"[OK] people parsed: {url} -> tables/{slug}.csv ({len(people)} rows)")

        # Соберём новые ссылки
        if depth < MAX_DEPTH:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not is_internal_link(href):
                    continue
                nxt = urljoin(url, href)
                nxt = urldefrag(nxt)[0]
                if urlparse(nxt).netloc != DOMAIN:
                    continue
                if nxt not in seen:
                    q.append(QueueItem(nxt, depth + 1))

    # Сводный CSV
    if all_people_rows:
        with open(ALL_PEOPLE_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_url", "title", "line_no", "settlement", "folio",
                        "role", "name", "patronymic", "family", "age", "raw_segment"])
            for row in all_people_rows:
                w.writerow(row)
        print(f"[OK] consolidated: {ALL_PEOPLE_PATH} ({len(all_people_rows)} rows)")

    print(f"Done. Saved pages: {saved_pages}, Parsed pages: {parsed_pages}")

if __name__ == "__main__":
    crawl()

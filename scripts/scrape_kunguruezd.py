#!/usr/bin/env python3
import json
import time
import sys
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://kunguruezd.myqip.ru/"

HEADERS = {
    "User-Agent": "GenealogyAggregatorBot/0.1 (+https://github.com/ant69)"
}

def get_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    # Поддержка старых кодировок: попробуем угадать
    if not resp.encoding or resp.encoding.lower() in ("ascii", "utf-8"):
        resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text
    return BeautifulSoup(html, "lxml")

def scrape_sections(index_url=BASE_URL):
    soup = get_soup(index_url)

    sections = []
    now_iso = datetime.now(timezone.utc).isoformat()

    # ВНИМАНИЕ: селекторы нужно подогнать под фактическую верстку форума.
    # Ниже — три «кандидата», раскомментируйте/адаптируйте тот, что реально сработает.
    candidates = [
        # пример: ссылки разделов внутри контейнера с классом "forum" или "category"
        ("a", {"class": "forumtitle"}),
        ("a", {"class": "catTitle"}),
        ("a", {}),
    ]

    seen = set()
    for tag_name, attrs in candidates:
        for a in soup.find_all(tag_name, attrs=attrs):
            href = (a.get("href") or "").strip()
            title = (a.get_text(strip=True) or "")
            if not href or not title:
                continue
            url = urljoin(index_url, href)
            key = (title, url)
            # Отфильтруем мусор и дубликаты
            if url.startswith(BASE_URL) and key not in seen:
                seen.add(key)
                sections.append({
                    "source": BASE_URL,
                    "section_title": title,
                    "section_url": url,
                    "topics_count": None,   # можно допарсить на странице раздела
                    "posts_count": None,    # см. комментарий выше
                    "last_post_dt": None,   # см. комментарий выше
                    "collected_at": now_iso
                })

    # вежливая пауза (если будем заходить в разделы)
    # time.sleep(1)

    return sections

def main():
    try:
        sections = scrape_sections()
        out_path = "data/forum_index.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(sections)} sections to {out_path}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

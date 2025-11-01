# Genealogy Aggregator (prototype)

Публичный прототип для агрегации открытых данных (форумы, реестры, справочники).
Данные сохраняются в `data/` и обновляются GitHub Actions.

## Полезные ссылки
- Сырые JSON: `data/*.json`
- (Опционально) Публикация на GitHub Pages: `https://ant69.github.io/genealogy-aggregator/`

## Как работает
1) `scripts/scrape_kunguruezd.py` собирает индекс разделов с форума kunguruezd.myqip.ru (минимально).
2) Скрипт записывает `data/forum_index.json`.
3) Workflow `scrape.yml` гоняет скрипт по расписанию и коммитит обновления.
4) Workflow `pages.yml` публикует содержимое `data/` на GitHub Pages.

#!/usr/bin/env bash
# Сборка PDF из docs/Salym_для_человека.md
# - pre-render mermaid блоков через mmdc
# - запуск md-to-pdf с кастомным CSS

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/docs/Salym_для_человека.md"
ASSETS_DIR="$ROOT/docs/assets"
TMP_DIR="$(mktemp -d)"
TMP_MD="$TMP_DIR/Salym_для_человека.md"
PUPPETEER_CFG="$TMP_DIR/puppeteer.json"

mkdir -p "$ASSETS_DIR"

# 1. Pre-render mermaid gantt (в docs/assets/)
cat > "$PUPPETEER_CFG" <<'JSON'
{"args":["--no-sandbox","--disable-setuid-sandbox"]}
JSON

cat > "$TMP_DIR/gantt_ws1333.mmd" <<'MMD'
gantt
    title WS-1333 · жизнь установок за 2015–2016
    dateFormat  YYYY-MM-DD
    axisFormat  %Y-%m

    section Циклы
    Цикл 1 — R=0 / Seal section        :done,  c1, 2015-03-06, 2015-05-24
    Цикл 2 — Клин / Pump section1      :crit,  c2, 2015-05-28, 2015-06-01
    Цикл 3 — Клин / Seal section       :done,  c3, 2015-06-10, 2016-02-10
    Цикл 4 — R=0 / Seal section        :done,  c4, 2016-02-15, 2016-06-05
    Цикл 5 — R=0 / Motor-top           :done,  c5, 2016-06-11, 2016-11-22
    Цикл 6 — Отсутствие подачи / —     :crit,  c6, 2016-12-11, 2016-12-14
MMD

mmdc -i "$TMP_DIR/gantt_ws1333.mmd" \
     -o "$ASSETS_DIR/salym_ws1333_gantt.png" \
     -w 1600 \
     -p "$PUPPETEER_CFG" >/dev/null

# 2. Скопировать MD и заменить mermaid-блок на ссылку на картинку
python3 - "$SRC" "$TMP_MD" "$ASSETS_DIR" <<'PY'
import re, sys
src_path, dst_path, assets_dir = sys.argv[1:4]
with open(src_path, encoding='utf-8') as f:
    text = f.read()
# Заменим весь mermaid-блок на изображение
new = re.sub(
    r'```mermaid\s+gantt[\s\S]+?```',
    '![WS-1333 — временная шкала циклов](assets/salym_ws1333_gantt.png)',
    text,
    count=1,
)
with open(dst_path, 'w', encoding='utf-8') as f:
    f.write(new)
PY

# 3. Временные assets для md-to-pdf (работает относительно cwd)
ln -sf "$ASSETS_DIR" "$TMP_DIR/assets"

# 4. CSS для md-to-pdf
cat > "$TMP_DIR/style.css" <<'CSS'
@page { size: A4; margin: 18mm 14mm; }
body {
  font-family: "DejaVu Sans", "Noto Sans", sans-serif;
  font-size: 10pt;
  line-height: 1.45;
  color: #1a1a1a;
}
h1 { font-size: 20pt; color: #111; border-bottom: 2px solid #333; padding-bottom: 4pt; margin-top: 18pt; page-break-after: avoid; }
h2 { font-size: 15pt; color: #222; margin-top: 16pt; page-break-after: avoid; }
h3 { font-size: 12.5pt; color: #333; margin-top: 12pt; page-break-after: avoid; }
h4 { font-size: 11pt; color: #444; margin-top: 10pt; page-break-after: avoid; }
p { margin: 4pt 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; width: 100%; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #aaa; padding: 4pt 6pt; vertical-align: top; text-align: left; }
th { background: #e8e8e8; font-weight: 600; }
tbody tr:nth-child(even) td { background: #f7f7f7; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; background: #f2f2f2; padding: 1pt 3pt; border-radius: 2pt; }
pre { background: #f5f5f5; padding: 8pt; font-size: 8.5pt; overflow-x: auto; border: 1px solid #ddd; border-radius: 3pt; page-break-inside: avoid; }
pre code { background: transparent; padding: 0; }
blockquote { border-left: 3px solid #999; padding-left: 8pt; color: #555; margin-left: 0; }
img { max-width: 100%; height: auto; display: block; margin: 8pt auto; }
ul, ol { margin: 4pt 0 4pt 18pt; padding: 0; }
li { margin: 2pt 0; }
CSS

# 5. Запуск md-to-pdf
OUT="$ROOT/docs/Salym_для_человека.pdf"
cd "$TMP_DIR"
md-to-pdf "$TMP_MD" --stylesheet "$TMP_DIR/style.css" --pdf-options '{"format":"A4","margin":{"top":"18mm","bottom":"18mm","left":"14mm","right":"14mm"},"printBackground":true}' >/dev/null

mv "$TMP_DIR/Salym_для_человека.pdf" "$OUT"

# 6. Cleanup
rm -rf "$TMP_DIR"

echo "PDF: $OUT"
echo "Asset: $ASSETS_DIR/salym_ws1333_gantt.png"
ls -la "$OUT"

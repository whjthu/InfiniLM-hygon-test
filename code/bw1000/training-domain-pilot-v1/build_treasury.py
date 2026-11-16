import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


SOURCE = 'https://fiscaldata.treasury.gov/datasets/debt-to-the-penny/debt-to-the-penny'
API = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny'
POLICY = 'https://fiscaldata.treasury.gov/data/about-us/901-1%20Open%20Data%20Policy.pdf'
FIELDS = ('record_date', 'debt_held_public_amt', 'intragov_hold_amt', 'tot_pub_debt_out_amt')
HEADERS = ('Date', 'Debt held by public', 'Intragov. holdings', 'Total public debt')
RANGES = {
    'train': (date(2022, 1, 1), date(2023, 12, 31), 72),
    'dev': (date(2024, 1, 1), date(2024, 12, 31), 36),
    'blind': (date(2026, 1, 1), date(2026, 6, 30), 18),
}
TOPICS = ('lookup', 'signed_difference', 'maximum_date')
AMOUNT = re.compile(r'[0-9]+\.[0-9]{2}', flags=re.ASCII)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def read_rows(raw):
    payload = json.loads(raw)
    records = payload['data']
    meta = payload['meta']
    if not records or meta['total-pages'] != 1 or meta['total-count'] != len(records):
        raise ValueError('Source must be one complete API result')
    grouped = defaultdict(list)
    seen = set()
    excluded_rows, excluded_identity_failures = 0, 0
    for record in records:
        day = date.fromisoformat(record['record_date'])
        if day in seen or not date(2022, 1, 1) <= day <= date(2026, 6, 30):
            raise ValueError(f'Duplicate or out-of-window date {day}')
        seen.add(day)
        values = [record[field] for field in FIELDS[1:]]
        if not all(isinstance(value, str) and AMOUNT.fullmatch(value) for value in values):
            raise ValueError(f'Invalid source amount on {day}')
        public, intra, total = map(Decimal, values)
        identity_failed = public <= 0 or intra <= 0 or public + intra != total
        if not any(start <= day <= end for start, end, _ in RANGES.values()):
            excluded_rows += 1
            excluded_identity_failures += identity_failed
            continue
        if identity_failed:
            raise ValueError(f'Debt identity failed on {day}')
        grouped[day.strftime('%Y-%m')].append((day.isoformat(), public, intra, total))
    return len(seen), excluded_rows, excluded_identity_failures, grouped


def split_tables(grouped):
    selected = {}
    for name, (start, end, quota) in RANGES.items():
        candidates = []
        for month, rows in sorted(grouped.items()):
            day = date.fromisoformat(month + '-01')
            if start <= day <= end:
                rows = sorted(rows)
                if len(rows) < 18:
                    raise ValueError(f'Missing business days in {month}')
                candidates.extend((month, rows[pos:pos + 5]) for pos in range(0, len(rows) - 4, 5))
        months = {(year, month) for year in range(start.year, end.year + 1)
                  for month in range(1, 13) if start <= date(year, month, 1) <= end}
        if len({month for month, _ in candidates}) != len(months) or len(candidates) < quota:
            raise ValueError(f'Incomplete month coverage or quota for {name}')
        # Keep three tables per month, spreading selection across the month before rendering.
        by_month = defaultdict(list)
        for month, rows in candidates:
            by_month[month].append(rows)
        if quota != 3 * len(months):
            raise ValueError(f'Three tables per month required for {name}')
        selected[name] = [rows for tables in by_month.values()
                          for rows in (tables[0], tables[len(tables) // 2], tables[-1])]
    date_sets = [{row[0] for table in tables for row in table} for tables in selected.values()]
    if len(set.union(*date_sets)) != sum(map(len, date_sets)):
        raise ValueError('Date leakage between splits')
    return selected


def draw_table(rows, font, path):
    image = Image.new('RGB', (1220, 280), '#ffffff')
    draw = ImageDraw.Draw(image)
    draw.text((24, 17), 'U.S. Treasury | Debt to the Penny | USD', fill='#152338', font=font)
    for index, row in enumerate((HEADERS, *rows)):
        y = 59 + 34 * index
        draw.rectangle((18, y - 3, 1200, y + 29), fill=('#dde9f5' if index == 0 else
                                                        '#f3f7fb' if index % 2 else '#ffffff'))
        for left, width, value in zip((24, 195, 530, 855), (171, 335, 325, 335), row):
            label = f'{value:,.2f}' if isinstance(value, Decimal) else value
            if draw.textlength(label, font=font) > width - 12:
                raise ValueError('Text would clip with selected font')
            draw.text((left, y), label, fill='#152338', font=font)
    image.save(path, optimize=False)


def questions(rows, index):
    label = ('debt held by the public', 'intragovernmental holdings', 'total public debt')[index % 3]
    col = 1 + index % 3
    chosen = rows[(index // 3) % len(rows)]
    yield TOPICS[0], f'What was {label} on {chosen[0]}? Return USD to two decimals.', f'{chosen[col]:.2f}'
    first, last = rows[0], rows[-1]
    yield (TOPICS[1], f'What is the signed change in {label} from {first[0]} to {last[0]} '
           '(last minus first)? Return USD to two decimals.', f'{last[col] - first[col]:.2f}')
    max_value = max(row[col] for row in rows)
    first_max = next(row[0] for row in rows if row[col] == max_value)
    yield (TOPICS[2], f'On which date was {label} highest in the table? If tied, choose the earliest '
           'date. Return YYYY-MM-DD.', first_max)


def write_jsonl(path, rows):
    raw = b''.join(canonical(row) + b'\n' for row in rows)
    path.write_bytes(raw)
    return sha(raw)


def build(source, expected_sha, font_file, output):
    raw = source.read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError('Input SHA-256 mismatch')
    observed_rows, excluded_rows, excluded_identity_failures, grouped = read_rows(raw)
    selected = split_tables(grouped)
    font = ImageFont.truetype(str(font_file), 24)
    output.mkdir()
    manifest = {'source': SOURCE, 'source_api': API, 'source_policy': POLICY,
                'source_SHA256': expected_sha, 'source_rows': observed_rows,
                'outside_split_rows': excluded_rows,
                'outside_split_identity_failures': excluded_identity_failures,
                'font_SHA256': sha(font_file.read_bytes()), 'pillow_version': pillow_version,
                'splits': {}, 'label_origin': 'Exact Decimal operations on the official JSON data'}
    for name, tables in selected.items():
        folder = output / name
        folder.mkdir()
        records, labels, image_pins = [], [], []
        for index, rows in enumerate(tables):
            table_id = rows[0][0] + '_' + rows[-1][0]
            image_path = folder / (table_id + '.png')
            draw_table(rows, font, image_path)
            image_pins.append([table_id, sha(image_path.read_bytes())])
            for topic, question, answer in questions(rows, index):
                identifier = f'{table_id}_{topic}'
                messages = [{'role': 'user', 'content': '<image>\n' + question}]
                if name == 'train':
                    messages.append({'role': 'assistant', 'content': answer})
                records.append({'messages': messages, 'images': [f'{name}/{image_path.name}']})
                labels.append({'id': identifier, 'topic': topic, 'answer': answer, 'table': table_id})
        manifest['splits'][name] = {
            'tables': len(tables), 'examples': len(records), 'examples_per_topic': len(tables),
            'topic_names': list(TOPICS), 'records_SHA256': write_jsonl(folder / 'inputs.jsonl', records),
            'labels_SHA256': write_jsonl(folder / 'labels.jsonl', labels),
            'image_inventory_SHA256': write_jsonl(folder / 'images.jsonl', image_pins),
        }
    (output / 'manifest.json').write_bytes(canonical(manifest) + b'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.source_sha256, args.font, args.output), indent=2))

#!/usr/bin/env python3
"""Validate CSE 151B Kaggle submission CSV against a JSONL input file."""
import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='private.jsonl or public.jsonl')
    ap.add_argument('--submission', required=True, help='submission.csv')
    args=ap.parse_args()
    rows=load_jsonl(args.input)
    expected=[int(r['id']) for r in rows]
    expected_set=set(expected)
    errors=[]
    with open(args.submission, encoding='utf-8', newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['id','response']:
            errors.append(f"header must be exactly ['id','response']; got {reader.fieldnames}")
            data=[]
        else:
            data=list(reader)
    got=[]
    for idx,row in enumerate(data, start=2):
        try:
            rid=int(row.get('id',''))
            got.append(rid)
        except Exception:
            errors.append(f'row {idx}: non-integer id {row.get("id")!r}')
            continue
        resp=row.get('response','')
        if not resp.strip():
            errors.append(f'row {idx} id={rid}: empty response')
        if '\\boxed{' not in resp:
            errors.append(f'row {idx} id={rid}: missing \\boxed{{...}}')
    got_set=set(got)
    missing=sorted(expected_set-got_set)
    extra=sorted(got_set-expected_set)
    dup=len(got)-len(got_set)
    if len(data)!=len(rows):
        errors.append(f'row count mismatch: csv={len(data)} expected={len(rows)}')
    if missing:
        errors.append(f'missing {len(missing)} ids, first 20: {missing[:20]}')
    if extra:
        errors.append(f'extra {len(extra)} ids, first 20: {extra[:20]}')
    if dup:
        errors.append(f'duplicate id rows: {dup}')
    if errors:
        print('INVALID')
        for e in errors[:50]: print('-', e)
        raise SystemExit(1)
    print(f'VALID: {args.submission} has {len(data)} rows and covers all ids in {args.input}')

if __name__=='__main__':
    main()

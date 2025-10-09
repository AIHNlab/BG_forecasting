"""Aggregate cg_ega_summary.csv files across experiment runs.

Searches under experiments/<TEST_FOLDER> for files named cg_ega_summary.csv and
computes mean AP, BE and EP per horizon (6,12,24) for each run. Writes:

- aggregated_cg_ega_by_tag_dataset.csv : one row per run with columns AP_6,BE_6,EP_6,...
- aggregated_tidy_cg_ega_stacked.csv : one row per run/horizon with AP,BE,EP columns

Usage: python scripts/aggregate_cg_ega.py [TEST_FOLDER]
If no TEST_FOLDER is provided the default is 'TransformerFinal'.
"""
from collections import OrderedDict
import os
import sys
import csv

# Default test folder: choose the experiments subfolder you want to aggregate.
TEST_FOLDER = "Geneva"#"FinalResultsFolder"#'Robustness'
if len(sys.argv) > 1 and sys.argv[1].strip():
    TEST_FOLDER = sys.argv[1].strip()

ROOT = os.path.join(os.getcwd(), 'experiments')
SEARCH_ROOT = os.path.join(ROOT, TEST_FOLDER) if TEST_FOLDER else ROOT

OUT_AGG = os.path.join(SEARCH_ROOT, 'aggregated_cg_ega_by_tag_dataset.csv')
OUT_STACK = os.path.join(SEARCH_ROOT, 'aggregated_tidy_cg_ega_stacked.csv')
OUT_STACK_BY_TAG = os.path.join(SEARCH_ROOT, 'aggregated_tidy_cg_ega_stacked_by_tag.csv')
OUT_STACK_BY_TAG_XLSX = os.path.join(SEARCH_ROOT, 'aggregated_tidy_cg_ega_stacked_by_tag.xlsx')

# Optional ordering of tags (comma-separated list can be provided via env or later extension)
ORDER_TAGS = []
# Optional highlight best (not implemented for AP/BE/EP by default) — kept for parity
HIGHLIGHT_BEST = False

def safe_float(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == '':
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_run_folder_name(name):
    info = {'dataset': None, 'tag': None}
    tokens = name.split('_')
    for t in tokens:
        if t.startswith('ds-'):
            info['dataset'] = t.split('-',1)[1]
        elif t.startswith('tag-'):
            info['tag'] = t.split('-',1)[1]
        elif t.startswith('ds') and '-' in t:
            info['dataset'] = t.split('-',1)[1]
    if info['tag'] is None:
        idx = name.find('_tag-')
        if idx != -1:
            part = name[idx+5:]
            if '_' in part:
                info['tag'] = part.split('_',1)[0]
            else:
                info['tag'] = part
    if info['tag'] is None:
        info['tag'] = ''
    return info


def aggregate_file(path):
    # sums[(horizon,metric)] and counts
    sums = OrderedDict()
    counts = OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # expect columns: Participant,Horizon,AP,BE,EP
            h = row.get('Horizon') or row.get('horizon')
            if h is None:
                continue
            h = str(h).strip()
            for metric in ('AP', 'BE', 'EP'):
                v = safe_float(row.get(metric))
                key = f'{metric}_{h}'
                if key not in sums:
                    sums[key] = 0.0
                    counts[key] = 0
                if v is not None:
                    sums[key] += v
                    counts[key] += 1

    means = {}
    for k in sums:
        if counts[k] > 0:
            means[k] = sums[k] / counts[k]
        else:
            means[k] = None
    return means


def find_cg_files(search_root):
    results = []
    if not os.path.exists(search_root):
        return results
    # Walk and find files named cg_ega_summary.csv
    for dirpath, dirnames, filenames in os.walk(search_root):
        if 'cg_ega_summary.csv' in filenames:
            full = os.path.join(dirpath, 'cg_ega_summary.csv')
            # infer base (the experiment folder under experiments/) and run folder name
            # assume dirpath .../<base>/<run_folder>/evaluation/
            # climb up two levels to get run_folder and base
            parts = dirpath.split(os.sep)
            if len(parts) >= 3:
                run_folder = parts[-2]
                base = parts[-3]
            else:
                run_folder = os.path.basename(dirpath)
                base = ''
            results.append((base, run_folder, full))
    return results


def main():
    files = find_cg_files(SEARCH_ROOT)
    if not files:
        print(f'No cg_ega_summary.csv files found under {SEARCH_ROOT!r}')
        return

    aggregated = []
    for base, run_folder, path in files:
        info = parse_run_folder_name(run_folder)
        means = aggregate_file(path)
        aggregated.append({'base': base, 'run_folder': run_folder, 'dataset': info['dataset'], 'tag': info['tag'], 'means': means})

    # collect all metric keys
    all_keys = set()
    for a in aggregated:
        all_keys.update(a['means'].keys())
    all_keys = sorted(all_keys)

    os.makedirs(os.path.dirname(OUT_AGG), exist_ok=True)
    with open(OUT_AGG, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['base', 'dataset', 'tag', 'run_folder'] + all_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in aggregated:
            row = {k: '' for k in fieldnames}
            row['base'] = a['base']
            row['dataset'] = a['dataset']
            row['tag'] = a['tag']
            row['run_folder'] = a['run_folder']
            for k in all_keys:
                v = a['means'].get(k)
                row[k] = '' if v is None else ('{:.6g}'.format(v))
            writer.writerow(row)

    print(f'Wrote aggregated CSV to: {OUT_AGG}')

    # write stacked tidy CSV: one row per run/horizon with AP,BE,EP
    stacked_fields = ['base', 'dataset', 'tag', 'run_folder', 'horizon', 'AP', 'BE', 'EP']
    with open(OUT_STACK, 'w', newline='', encoding='utf-8') as sf:
        sw = csv.DictWriter(sf, fieldnames=stacked_fields)
        sw.writeheader()
        for a in aggregated:
            for h in ['6', '12', '24']:
                row = {k: '' for k in stacked_fields}
                row['base'] = a['base']
                row['dataset'] = a['dataset']
                row['tag'] = a['tag']
                row['run_folder'] = a['run_folder']
                row['horizon'] = h
                for m in ['AP', 'BE', 'EP']:
                    key = f'{m}_{h}'
                    v = a['means'].get(key)
                    row[m] = '' if v is None else ('{:.6g}'.format(v))
                sw.writerow(row)

    print(f'Wrote stacked tidy CSV to: {OUT_STACK}')

    # --- write stacked-by-tag CSV ---
    def write_stacked_by_tag(aggregated):
        # build mapping dataset -> {tags: {tag: means}}
        tags = []
        mapping = {}
        for a in aggregated:
            t = a['tag'] or '(no-tag)'
            d = a['dataset'] or a['run_folder']
            if t not in tags:
                tags.append(t)
            if d not in mapping:
                mapping[d] = {'base': a['base'], 'tags': {}}
            mapping[d]['tags'][t] = a['means']

        if ORDER_TAGS:
            tags = [t for t in ORDER_TAGS if t in tags] + [t for t in sorted(tags) if t not in (ORDER_TAGS or [])]
        else:
            tags = sorted(tags)

        datasets = sorted(mapping.keys())

        # write CSV with two header rows: first row tag names repeated 3 times, second row AP/BE/EP
        with open(OUT_STACK_BY_TAG, 'w', newline='', encoding='utf-8') as of:
            writer = csv.writer(of)
            first_row = ['dataset', 'horizon'] + [t for t in tags for _ in (0,1,2)]
            second_row = ['', ''] + ['AP', 'BE', 'EP'] * len(tags)
            writer.writerow(first_row)
            writer.writerow(second_row)

            for d in datasets:
                tagmap = mapping[d]['tags']
                for h in ['6', '12', '24']:
                    row = [d, h]
                    for t in tags:
                        means = tagmap.get(t, {})
                        for m in ['AP', 'BE', 'EP']:
                            key = f'{m}_{h}'
                            v = means.get(key)
                            row.append('' if v is None else ('{:.6g}'.format(v)))
                    writer.writerow(row)

        print(f'Wrote stacked-by-tag CSV to: {OUT_STACK_BY_TAG}')

    write_stacked_by_tag(aggregated)

    # --- write Excel with merged headers like aggregate_rmse.py ---
    def write_stacked_by_tag_excel(aggregated):
        try:
            from openpyxl import Workbook
            from openpyxl.utils import get_column_letter
            from openpyxl.styles import Border, Side, Alignment, Font
        except Exception:
            print('openpyxl not installed — skipping Excel output for stacked-by-tag. Install with: pip install openpyxl')
            return

        # build mapping dataset -> {base, tags}
        tags = []
        mapping = {}
        for a in aggregated:
            t = a['tag'] or '(no-tag)'
            d = a['dataset'] or a['run_folder']
            if t not in tags:
                tags.append(t)
            if d not in mapping:
                mapping[d] = {'base': a['base'], 'tags': {}}
            mapping[d]['tags'][t] = a['means']

        if ORDER_TAGS:
            tags = [t for t in ORDER_TAGS if t in tags] + [t for t in sorted(tags) if t not in (ORDER_TAGS or [])]
        else:
            tags = sorted(tags)
        datasets = sorted(mapping.keys())

        wb = Workbook()
        ws = wb.active
        ws.title = 'stacked_by_tag'

        # first header row: dataset,horizon then tag names merged across 3 columns
        for col, val in enumerate(['dataset', 'horizon'], start=1):
            ws.cell(row=1, column=col, value=val)
        start_col = 3
        for i, t in enumerate(tags):
            c1 = start_col + i*3
            c2 = c1 + 2
            ws.merge_cells(start_row=1, start_column=c1, end_row=1, end_column=c2)
            ws.cell(row=1, column=c1, value=str(t))
            # second row: AP, BE, EP
            ws.cell(row=2, column=c1, value='AP')
            ws.cell(row=2, column=c1+1, value='BE')
            ws.cell(row=2, column=c1+2, value='EP')

        # write data rows starting at row 3; merge dataset cells across 3 horizons
        out_row = 3
        for d in datasets:
            tagmap = mapping[d]['tags']
            merge_start = out_row
            merge_end = out_row + 3 - 1
            ws.cell(row=merge_start, column=1, value=d)
            ws.merge_cells(start_row=merge_start, start_column=1, end_row=merge_end, end_column=1)
            for h in ['6', '12', '24']:
                ws.cell(row=out_row, column=2, value=h)
                # write tag columns
                for i, t in enumerate(tags):
                    means = tagmap.get(t, {})
                    for j, m in enumerate(['AP', 'BE', 'EP']):
                        key = f'{m}_{h}'
                        v = means.get(key)
                        col = 3 + i*3 + j
                        ws.cell(row=out_row, column=col, value=None if v is None else float(v))
                out_row += 1

        # adjust widths
        last_col = start_col + len(tags)*3 - 1
        for col in range(1, last_col + 1):
            ws.column_dimensions[get_column_letter(col)].width = 15

        # styles
        thin = Side(border_style="thin", color="000000")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)
        header_font = Font(bold=True)
        center = Alignment(horizontal='center', vertical='center')

        # apply header styles
        for col in range(1, last_col + 1):
            cell1 = ws.cell(row=1, column=col)
            cell2 = ws.cell(row=2, column=col)
            try:
                cell1.font = header_font
                cell2.font = header_font
                cell1.alignment = center
                cell2.alignment = center
                cell1.border = border
                cell2.border = border
            except Exception:
                pass

        # apply border to data cells
        for r in range(3, out_row):
            for c in range(1, last_col + 1):
                try:
                    ws.cell(row=r, column=c).border = border
                    ws.cell(row=r, column=c).alignment = Alignment(horizontal='center')
                except Exception:
                    pass

        # make dataset names bold
        for idx, d in enumerate(datasets):
            merge_start = 3 + idx * 3
            try:
                ws.cell(row=merge_start, column=1).font = header_font
            except Exception:
                pass

        wb.save(OUT_STACK_BY_TAG_XLSX)
        print(f'Wrote stacked-by-tag Excel to: {OUT_STACK_BY_TAG_XLSX}')

    write_stacked_by_tag_excel(aggregated)


if __name__ == '__main__':
    main()

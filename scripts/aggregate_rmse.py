import os
import os
import csv
import math
import sys
from collections import defaultdict, OrderedDict

# Set TEST_FOLDER to the experiment subfolder to aggregate (e.g. 'FinalLinReg').
# If an argument is provided on the command line it will override this value.
TEST_FOLDER = "Geneva"#'FinalResultsFolder'
if len(sys.argv) > 1 and sys.argv[1].strip():
    TEST_FOLDER = sys.argv[1].strip()

# Optional second argument: comma-separated tag order, e.g. "2304,888,312,96"
ORDER_TAGS = ["Transformer","BestHpLstm","LstmBig","LinReg"]
#ORDER_TAGS = ["2304,888,312,96"]
if len(sys.argv) > 2 and sys.argv[2].strip():
    ORDER_TAGS = [t.strip() for t in sys.argv[2].split(',') if t.strip()]

# Optional third argument: enable highlighting best model (lowest RMSE) in Excel
HIGHLIGHT_BEST = True
if len(sys.argv) > 3 and sys.argv[3].strip():
    v = sys.argv[3].strip().lower()
    if v in ('highlight-best', 'highlight_best', 'best', 'bold-best', 'bold_best', 'highlight'):
        HIGHLIGHT_BEST = True

# Search root is experiments/<TEST_FOLDER>
ROOT = os.path.join(os.getcwd(), 'experiments')
SEARCH_ROOT = os.path.join(ROOT, TEST_FOLDER) if TEST_FOLDER else ROOT
OUT_CSV = os.path.join(SEARCH_ROOT, 'aggregated_rmse_by_tag_dataset.csv')
def safe_float(s):
    if s is None:
        return None
    s = s.strip()
    if s == '':
        return None
    try:
        return float(s)
    except:
        return None


def parse_run_folder_name(name):
    # tokens like run_ds-Ohio_tag-LinReg or run_ds-Colas_tag-LinReg
    tokens = name.split('_')
    info = {'dataset': None, 'tag': None}
    for t in tokens:
        if t.startswith('ds-'):
            info['dataset'] = t.split('-',1)[1]
        elif t.startswith('tag-'):
            info['tag'] = t.split('-',1)[1]
        elif t.startswith('ds') and '-' in t:
            info['dataset'] = t.split('-',1)[1]
    # fallback: try to find pattern 'ds' inside name
    if info['dataset'] is None:
        # try to find after 'run_' prefix
        if name.startswith('run_'):
            rest = name[4:]
            # take up to first '_'
            if '_' in rest:
                info['dataset'] = rest.split('_',1)[0].replace('ds-','')
            else:
                info['dataset'] = rest
    if info['tag'] is None:
        # try regex-like scan for '_tag-'
        idx = name.find('_tag-')
        if idx != -1:
            part = name[idx+5:]
            if '_' in part:
                info['tag'] = part.split('_',1)[0]
            else:
                info['tag'] = part
    if info['tag'] is None:
        # as last fallback, set tag to empty string
        info['tag'] = ''
    return info


def aggregate_file(path):
    sums = OrderedDict()
    counts = OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k,v in row.items():
                val = safe_float(v)
                if val is None:
                    continue
                if k not in sums:
                    sums[k] = 0.0
                    counts[k] = 0
                sums[k] += val
                counts[k] += 1
    means = {}
    for k in sums:
        if counts[k] > 0:
            means[k] = sums[k] / counts[k]
        else:
            means[k] = None
    return means


def find_rmse_files(root, test_folder=None):
    results = []
    # If a specific test_folder is given, root points to experiments/<test_folder>
    if test_folder:
        base = test_folder
        for entry in os.listdir(root):
            run_dir = os.path.join(root, entry)
            if not os.path.isdir(run_dir):
                continue
            eval_path = os.path.join(run_dir, 'evaluation', 'rmse_summary.csv')
            if os.path.exists(eval_path):
                results.append((base, entry, eval_path))
        return results

    # Otherwise keep original behavior: experiments/*/run_*/evaluation/rmse_summary.csv
    for base in os.listdir(root):
        base_path = os.path.join(root, base)
        if not os.path.isdir(base_path):
            continue
        for entry in os.listdir(base_path):
            run_dir = os.path.join(base_path, entry)
            if not os.path.isdir(run_dir):
                continue
            eval_path = os.path.join(run_dir, 'evaluation', 'rmse_summary.csv')
            if os.path.exists(eval_path):
                results.append((base, entry, eval_path))
    return results


def main():
    files = find_rmse_files(SEARCH_ROOT, TEST_FOLDER)
    if not files:
        print(f'No rmse_summary.csv files found under {SEARCH_ROOT!r}')
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

    # write full aggregated CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
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
                row[k] = '' if v is None else f"{v:.6f}"
            writer.writerow(row)

    print(f'Wrote aggregated CSV to: {OUT_CSV}\n')

    # produce per-region tidy CSVs (normo/hypo/hyper) that contain both RMSE and MAE columns
    def write_region_tidy(suffixes):
        for sfx in suffixes:
            out_name = os.path.join(SEARCH_ROOT, f'aggregated_{sfx}.csv')
            with open(out_name, 'w', newline='', encoding='utf-8') as of:
                fn = ['base', 'dataset', 'tag', 'run_folder', 'RMSE_6', 'RMSE_12', 'RMSE_24', 'MAE_6', 'MAE_12', 'MAE_24']
                w = csv.DictWriter(of, fieldnames=fn)
                w.writeheader()
                for a in aggregated:
                    row = {k: '' for k in fn}
                    row['base'] = a['base']
                    row['dataset'] = a['dataset']
                    row['tag'] = a['tag']
                    row['run_folder'] = a['run_folder']
                    # RMSE_normo_6 etc. are the keys inside a['means']
                    for metric in ['RMSE', 'MAE']:
                        for h in ['6', '12', '24']:
                            key = f"{metric}_{sfx}_{h}"
                            out_key = f"{metric}_{h}"
                            v = a['means'].get(key)
                            row[out_key] = '' if v is None else f"{v:.6f}"
                    w.writerow(row)
            print(f'Wrote {out_name}')

    suffixes = ['normo', 'hypo', 'hyper']
    write_region_tidy(suffixes)

    # also write a tidy CSV with RMSE_6/12/24 and MAE_6/12/24 as columns for each tag/dataset
    tidy_out = os.path.join(SEARCH_ROOT, 'aggregated_tidy_rmse_mae.csv')
    tidy_fields = ['base', 'dataset', 'tag', 'run_folder', 'RMSE_6', 'RMSE_12', 'RMSE_24', 'MAE_6', 'MAE_12', 'MAE_24']
    with open(tidy_out, 'w', newline='', encoding='utf-8') as tf:
        tw = csv.DictWriter(tf, fieldnames=tidy_fields)
        tw.writeheader()
        for a in aggregated:
            row = {k: '' for k in tidy_fields}
            row['base'] = a['base']
            row['dataset'] = a['dataset']
            row['tag'] = a['tag']
            row['run_folder'] = a['run_folder']
            for m in ['RMSE_6', 'RMSE_12', 'RMSE_24', 'MAE_6', 'MAE_12', 'MAE_24']:
                v = a['means'].get(m)
                row[m] = '' if v is None else f"{v:.6f}"
            tw.writerow(row)
    print(f'Wrote tidy CSV to: {tidy_out}\n')

    # --- New: write a stacked tidy CSV (one row per horizon) ---
    # This produces rows with columns: base, dataset, tag, run_folder, horizon, RMSE, MAE
    stacked_out = os.path.join(SEARCH_ROOT, 'aggregated_tidy_rmse_mae_stacked.csv')
    stacked_fields = ['base', 'dataset', 'tag', 'run_folder', 'horizon', 'RMSE', 'MAE']
    with open(stacked_out, 'w', newline='', encoding='utf-8') as sf:
        sw = csv.DictWriter(sf, fieldnames=stacked_fields)
        sw.writeheader()
        for a in aggregated:
            base = a['base']
            dataset = a['dataset']
            tag = a['tag']
            run_folder = a['run_folder']
            for h in ['6', '12', '24']:
                row = {k: '' for k in stacked_fields}
                row['base'] = base
                row['dataset'] = dataset
                row['tag'] = tag
                row['run_folder'] = run_folder
                row['horizon'] = h
                rmse_key = f'RMSE_{h}'
                mae_key = f'MAE_{h}'
                v_rmse = a['means'].get(rmse_key)
                v_mae = a['means'].get(mae_key)
                row['RMSE'] = '' if v_rmse is None else f"{v_rmse:.6f}"
                row['MAE'] = '' if v_mae is None else f"{v_mae:.6f}"
                sw.writerow(row)
    print(f'Wrote stacked tidy CSV to: {stacked_out}\n')

    # --- New: write a stacked CSV where tags are separate columns ---
    # For each dataset/run_folder and horizon produce one row. Columns: base, dataset, run_folder, horizon,
    # then for each tag two columns: <tag>_RMSE and <tag>_MAE
    def write_stacked_by_tag(aggregated):
        # collect tags and mapping dataset -> {tag: means, base}
        tags = []
        mapping = {}
        for a in aggregated:
            t = a['tag'] or '(no-tag)'
            d = a['dataset'] or a['run_folder']
            if t not in tags:
                tags.append(t)
            if d not in mapping:
                mapping[d] = {'base': a['base'], 'tags': {}}
            # if multiple runs for same dataset+tag exist, last one wins (keeps behavior simple)
            mapping[d]['tags'][t] = a['means']

        if ORDER_TAGS:
            tags = [t for t in ORDER_TAGS if t in tags] + [t for t in sorted(tags) if t not in (ORDER_TAGS or [])]
        else:
            tags = sorted(tags)
        datasets = sorted(mapping.keys())

        out_path = os.path.join(SEARCH_ROOT, 'aggregated_tidy_rmse_mae_stacked_by_tag.csv')

        # Build two-row header: first row = tags repeated (one tag per two columns),
        # second row = metric names (RMSE, MAE) repeated for each tag.
        # Data rows: base, dataset, horizon, then for each tag RMSE then MAE.
        with open(out_path, 'w', newline='', encoding='utf-8') as of:
            writer = csv.writer(of)
            # First header row: dataset,horizon then tag names repeated twice
            first_row = ['dataset', 'horizon'] + [t for t in tags for _ in (0, 1)]
            # Second header row: empty cells for initial columns, then RMSE/MAE for each tag
            second_row = ['', ''] + ['RMSE', 'MAE'] * len(tags)
            writer.writerow(first_row)
            writer.writerow(second_row)

            # Write data rows in same order (no base column)
            for d in datasets:
                tagmap = mapping[d]['tags']
                for h in ['6', '12', '24']:
                    row = [d, h]
                    for t in tags:
                        means = tagmap.get(t)
                        rmse_key = f"RMSE_{h}"
                        mae_key = f"MAE_{h}"
                        v_rmse = means.get(rmse_key) if means is not None else None
                        v_mae = means.get(mae_key) if means is not None else None
                        row.append('' if v_rmse is None else f"{v_rmse:.6f}")
                        row.append('' if v_mae is None else f"{v_mae:.6f}")
                    writer.writerow(row)

        print(f'Wrote stacked-by-tag CSV to: {out_path}\n')

    write_stacked_by_tag(aggregated)

    # Also write an Excel (.xlsx) version with merged header cells so the repeated tag names
    # appear as a single merged cell spanning RMSE and MAE columns.
    def write_stacked_by_tag_excel(aggregated):
        try:
            from openpyxl import Workbook
            from openpyxl.utils import get_column_letter
            from openpyxl.styles import Border, Side, Alignment, Font
        except Exception:
            print('openpyxl not installed — skipping Excel output for stacked-by-tag.\nInstall with: pip install openpyxl')
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

        out_xlsx = os.path.join(SEARCH_ROOT, 'aggregated_tidy_rmse_mae_stacked_by_tag.xlsx')

        wb = Workbook()
        ws = wb.active
        ws.title = 'stacked_by_tag'

        # Write first header row: dataset,horizon then tag names merged across 2 columns
        # Columns 1..2 are dataset,horizon
        for col, val in enumerate(['dataset', 'horizon'], start=1):
            ws.cell(row=1, column=col, value=val)
        # start tag columns at col 3
        start_col = 3
        for i, t in enumerate(tags):
            c1 = start_col + i*2
            c2 = c1 + 1
            # merge the two header cells for the tag
            ws.merge_cells(start_row=1, start_column=c1, end_row=1, end_column=c2)
            ws.cell(row=1, column=c1, value=str(t))
            # second header row: RMSE, MAE
            ws.cell(row=2, column=c1, value='RMSE')
            ws.cell(row=2, column=c2, value='MAE')

    # write data rows starting at row 3 and merge dataset cells across the three horizons
        out_row = 3
        for d in datasets:
            tagmap = mapping[d]['tags']
            # reserve rows for three horizons and write dataset in top-left before merging
            merge_start = out_row
            merge_end = out_row + 3 - 1
            ws.cell(row=merge_start, column=1, value=d)
            ws.merge_cells(start_row=merge_start, start_column=1, end_row=merge_end, end_column=1)
            for h in ['6', '12', '24']:
                # only write horizon in column 2 for each row (dataset cell is merged)
                ws.cell(row=out_row, column=2, value=h)
                for i, t in enumerate(tags):
                    means = tagmap.get(t)
                    rmse_key = f"RMSE_{h}"
                    mae_key = f"MAE_{h}"
                    v_rmse = means.get(rmse_key) if means is not None else None
                    v_mae = means.get(mae_key) if means is not None else None
                    c1 = start_col + i*2
                    c2 = c1 + 1
                    if v_rmse is not None:
                        try:
                            cell_rm = ws.cell(row=out_row, column=c1, value=float(v_rmse))
                            cell_rm.number_format = '0.000'
                        except Exception:
                            # fallback: write as plain string
                            ws.cell(row=out_row, column=c1, value=f"{v_rmse:.3f}")
                    if v_mae is not None:
                        try:
                            cell_ma = ws.cell(row=out_row, column=c2, value=float(v_mae))
                            cell_ma.number_format = '0.000'
                        except Exception:
                            ws.cell(row=out_row, column=c2, value=f"{v_mae:.3f}")
                out_row += 1

        # Optionally adjust column widths a bit
        last_col = start_col + len(tags)*2 - 1
        for col in range(1, last_col + 1):
            ws.column_dimensions[get_column_letter(col)].width = 15

        # Apply borders and header styling
        thin = Side(border_style="thin", color="000000")
        thick = Side(border_style="medium", color="000000")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)
        header_font = Font(bold=True)
        center = Alignment(horizontal='center', vertical='center')

        # Apply style to header rows (1 and 2)
        for col in range(1, last_col + 1):
            cell1 = ws.cell(row=1, column=col)
            cell2 = ws.cell(row=2, column=col)
            try:
                cell1.font = header_font
                cell1.alignment = center
            except Exception:
                pass
            try:
                cell2.font = header_font
                cell2.alignment = center
            except Exception:
                pass
            try:
                cell1.border = border
            except Exception:
                pass
            try:
                cell2.border = border
            except Exception:
                pass

        # Apply border and alignment to all data cells
        for r in range(3, out_row):
            for c in range(1, last_col + 1):
                cell = ws.cell(row=r, column=c)
                try:
                    # center numeric/horizon cells, left-align dataset column
                    if c == 1:
                        cell.alignment = Alignment(horizontal='left', vertical='center')
                    else:
                        cell.alignment = center
                    cell.border = border
                except Exception:
                    pass

        # Make dataset names bold and add a thicker bottom border between dataset blocks
        for idx, d in enumerate(datasets):
            merge_start = 3 + idx * 3
            merge_end = merge_start + 3 - 1
            # bold the merged dataset cell (top-left)
            try:
                top_cell = ws.cell(row=merge_start, column=1)
                top_cell.font = Font(bold=True)
                top_cell.alignment = Alignment(horizontal='left', vertical='center')
            except Exception:
                pass
            # add a medium bottom border across the entire row to separate dataset blocks
            for c in range(1, last_col + 1):
                try:
                    cur = ws.cell(row=merge_end, column=c)
                    # preserve existing left/right/top but set bottom to thick
                    cur.border = Border(
                        left=thin, right=thin, top=thin, bottom=thick
                    )
                except Exception:
                    pass

        # If requested, find best (lowest) and second-best RMSE/MAE per dataset+horizon
        # and style the corresponding cells: best = bold + red, second-best = underline + blue.
        if HIGHLIGHT_BEST:
            best_map_rmse = {}   # (dataset,h) -> best_tag_for_rmse
            second_map_rmse = {} # (dataset,h) -> second_best_tag_for_rmse
            best_map_mae = {}    # (dataset,h) -> best_tag_for_mae
            second_map_mae = {}  # (dataset,h) -> second_best_tag_for_mae
            for d in datasets:
                tagmap = mapping[d]['tags']
                for h in ['6', '12', '24']:
                    # RMSE: build sorted list of (tag, value)
                    vals = []
                    for t in tags:
                        means = tagmap.get(t)
                        if not means:
                            continue
                        v = means.get(f"RMSE_{h}")
                        if v is None:
                            continue
                        vals.append((t, v))
                    vals.sort(key=lambda x: x[1])
                    if len(vals) >= 1:
                        best_map_rmse[(d, h)] = vals[0][0]
                    if len(vals) >= 2:
                        second_map_rmse[(d, h)] = vals[1][0]

                    # MAE: build sorted list of (tag, value)
                    vals = []
                    for t in tags:
                        means = tagmap.get(t)
                        if not means:
                            continue
                        v = means.get(f"MAE_{h}")
                        if v is None:
                            continue
                        vals.append((t, v))
                    vals.sort(key=lambda x: x[1])
                    if len(vals) >= 1:
                        best_map_mae[(d, h)] = vals[0][0]
                    if len(vals) >= 2:
                        second_map_mae[(d, h)] = vals[1][0]

            # helper to resolve merged dataset cell values by walking upward
            def get_dataset_for_row(r):
                val = ws.cell(row=r, column=1).value
                if val is not None and val != '':
                    return val
                rr = r - 1
                while rr >= 1:
                    v = ws.cell(row=rr, column=1).value
                    if v is not None and v != '':
                        return v
                    rr -= 1
                return None

            for row_idx in range(3, out_row):
                ds = get_dataset_for_row(row_idx)
                if ds is None:
                    continue
                h = str(ws.cell(row=row_idx, column=2).value)

                # RMSE best and second-best tags
                best_tag_rmse = best_map_rmse.get((ds, h))
                second_tag_rmse = second_map_rmse.get((ds, h))
                if best_tag_rmse:
                    try:
                        tag_idx = tags.index(best_tag_rmse)
                    except ValueError:
                        tag_idx = None
                    if tag_idx is not None:
                        rmse_col = start_col + tag_idx * 2
                        try:
                            ws.cell(row=row_idx, column=rmse_col).font = Font(bold=True, color='FF0000')
                        except Exception:
                            pass
                if second_tag_rmse and second_tag_rmse != best_tag_rmse:
                    try:
                        tag_idx2 = tags.index(second_tag_rmse)
                    except ValueError:
                        tag_idx2 = None
                    if tag_idx2 is not None:
                        rmse_col2 = start_col + tag_idx2 * 2
                        try:
                            ws.cell(row=row_idx, column=rmse_col2).font = Font(underline='single', color='0000FF')
                        except Exception:
                            pass

                # MAE best and second-best tags
                best_tag_mae = best_map_mae.get((ds, h))
                second_tag_mae = second_map_mae.get((ds, h))
                if best_tag_mae:
                    try:
                        tag_idx = tags.index(best_tag_mae)
                    except ValueError:
                        tag_idx = None
                    if tag_idx is not None:
                        mae_col = start_col + tag_idx * 2 + 1
                        try:
                            ws.cell(row=row_idx, column=mae_col).font = Font(bold=True, color='FF0000')
                        except Exception:
                            pass
                if second_tag_mae and second_tag_mae != best_tag_mae:
                    try:
                        tag_idx2 = tags.index(second_tag_mae)
                    except ValueError:
                        tag_idx2 = None
                    if tag_idx2 is not None:
                        mae_col2 = start_col + tag_idx2 * 2 + 1
                        try:
                            ws.cell(row=row_idx, column=mae_col2).font = Font(underline='single', color='0000FF')
                        except Exception:
                            pass

        wb.save(out_xlsx)
        print(f'Wrote stacked-by-tag Excel to: {out_xlsx}\n')

    write_stacked_by_tag_excel(aggregated)

    # write a pivot-style CSV where columns are grouped by model/tag and contain RMSE/MAE at horizons
    def write_pivot_by_model(aggregated):
        # collect tags and datasets
        tags = []
        datasets = set()
        # map (tag, dataset) -> means
        mapping = {}
        for a in aggregated:
            t = a['tag'] or '(no-tag)'
            d = a['dataset'] or a['run_folder']
            if t not in tags:
                tags.append(t)
            datasets.add(d)
            mapping[(t, d)] = a['means']

        if ORDER_TAGS:
            tags = [t for t in ORDER_TAGS if t in tags] + [t for t in sorted(tags) if t not in (ORDER_TAGS or [])]
        else:
            tags = sorted(tags)
        datasets = sorted(datasets)

        pivot_out = os.path.join(SEARCH_ROOT, 'aggregated_pivot_by_model.csv')
        with open(pivot_out, 'w', newline='', encoding='utf-8') as pf:
            writer = csv.writer(pf)
            # first header row: model names repeated
            first_row = ['dataset']
            for t in tags:
                first_row.extend([t] * 6)
            writer.writerow(first_row)
            # second header row: metric names per model
            metrics = ['RMSE_6', 'MAE_6', 'RMSE_12', 'MAE_12', 'RMSE_24', 'MAE_24']
            second_row = ['dataset']
            for _ in tags:
                second_row.extend(metrics)
            writer.writerow(second_row)

            # data rows
            for d in datasets:
                row = [d]
                for t in tags:
                    means = mapping.get((t, d), {})
                    for m in metrics:
                        v = means.get(m)
                        row.append('' if v is None else f"{v:.6f}")
                writer.writerow(row)

        print(f'Wrote pivot CSV to: {pivot_out}')

    write_pivot_by_model(aggregated)

    # write pivot CSVs for each region (normo/hypo/hyper) using region-specific metrics
    def write_pivot_by_region(aggregated, region):
        tags = []
        datasets = set()
        mapping = {}
        for a in aggregated:
            t = a['tag'] or '(no-tag)'
            d = a['dataset'] or a['run_folder']
            if t not in tags:
                tags.append(t)
            datasets.add(d)
            mapping[(t, d)] = a['means']

        tags = sorted(tags)
        datasets = sorted(datasets)

        pivot_out = os.path.join(SEARCH_ROOT, f'aggregated_pivot_{region}_by_model.csv')
        with open(pivot_out, 'w', newline='', encoding='utf-8') as pf:
            writer = csv.writer(pf)
            # first header row: model names repeated
            first_row = ['dataset']
            for t in tags:
                first_row.extend([t] * 6)
            writer.writerow(first_row)
            # second header row: metric names per model (region-specific values will be placed below)
            metrics = ['RMSE_6', 'MAE_6', 'RMSE_12', 'MAE_12', 'RMSE_24', 'MAE_24']
            second_row = ['dataset']
            for _ in tags:
                second_row.extend(metrics)
            writer.writerow(second_row)

            # data rows
            for d in datasets:
                row = [d]
                for t in tags:
                    means = mapping.get((t, d), {})
                    for metric in ['RMSE', 'MAE']:
                        for h in ['6', '12', '24']:
                            # region-specific key e.g. RMSE_normo_6
                            key = f"{metric}_{region}_{h}"
                            v = means.get(key)
                            row.append('' if v is None else f"{v:.6f}")
                writer.writerow(row)

        print(f'Wrote region pivot CSV to: {pivot_out}')

    for region in ['normo', 'hypo', 'hyper']:
        write_pivot_by_region(aggregated, region)

    # print a pivot table of RMSE_6, RMSE_12, RMSE_24 with tag rows and dataset columns
    metrics_to_show = ['RMSE_6', 'RMSE_12', 'RMSE_24']
    pivot = defaultdict(lambda: {})  # pivot[tag][dataset] = {metric: value}
    tags = set()
    datasets = set()
    for a in aggregated:
        t = a['tag'] or '(no-tag)'
        d = a['dataset'] or a['run_folder']
        tags.add(t)
        datasets.add(d)
        for m in metrics_to_show:
            pivot[t].setdefault(d, {})[m] = float(a['means'].get(m, float('nan'))) if a['means'].get(m) is not None else float('nan')

    if ORDER_TAGS:
        tags = [t for t in ORDER_TAGS if t in tags] + [t for t in sorted(tags) if t not in (ORDER_TAGS or [])]
    else:
        tags = sorted(tags)
    datasets = sorted(datasets)

    # print header
    hdr = ['tag/dataset'] + datasets
    print('\t'.join(hdr))
    for t in tags:
        row = [t]
        for d in datasets:
            vals = pivot[t].get(d, {})
            # choose RMSE_6 as representative value; format three metrics joined
            if d in pivot[t]:
                v6 = vals.get('RMSE_6', float('nan'))
                v12 = vals.get('RMSE_12', float('nan'))
                v24 = vals.get('RMSE_24', float('nan'))
                cell = f"6:{v6:.2f},12:{v12:.2f},24:{v24:.2f}"
            else:
                cell = ''
            row.append(cell)
        print('\t'.join(row))

    print('\nDone.')

if __name__ == '__main__':
    main()

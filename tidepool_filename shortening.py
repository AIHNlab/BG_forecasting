import os

def check_unique_prefix(folder_path, prefix_len=5, ignore_prefix="train_"):
    """
    Checks whether all files in a folder have unique prefixes of given length,
    ignoring a fixed starting substring (e.g., 'test_').
    
    Args:
        folder_path (str): Path to the folder.
        prefix_len (int): Length of the prefix to check (default: 3).
        ignore_prefix (str): Part of the filename to skip (default: "test_").
        
    Returns:
        bool: True if all prefixes are unique, False otherwise.
        dict: Mapping of prefix -> list of files with that prefix.
    """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    prefix_map = {}
    for f in files:
        if not f.startswith(ignore_prefix):
            continue  # skip files without the expected prefix
        stripped = f[len(ignore_prefix):]  # remove the "test_" part
        prefix = stripped[:prefix_len]
        prefix_map.setdefault(prefix, []).append(f)
    
    # Check uniqueness
    duplicates = {k: v for k, v in prefix_map.items() if len(v) > 1}
    
    if duplicates:
        print("⚠️ Found duplicates:")
        for prefix, file_list in duplicates.items():
            print(f"Prefix '{prefix}' -> {file_list}")
        return False, duplicates
    else:
        print("✅ All prefixes are unique.")
        return True, prefix_map
# Example usage:
folder = r"C:\Users\knutj\Code\BG_forecasting\standardized_datasets\Tidepool_SAP100\train"  # <-- change this
check_unique_prefix(folder)
#exit()

def shorten_filenames(folder_path, prefix_len=5, ignore_prefix="train_", dry_run=True):
    """Shorten filenames in a folder so that names keep only the first `prefix_len` characters
    after `ignore_prefix`.

    Behavior:
    - Only files starting with `ignore_prefix` are considered.
    - Target name will be: <ignore_prefix><first_prefix_len_chars><original_extension>
    - If multiple files would map to the same target, a numeric suffix `-1`, `-2`, ...
      is appended to make targets unique.
    - Performs a two-step rename (move to temporary names then to final names) to avoid
      accidental overwrites. Use `dry_run=True` to only print the planned changes.

    Returns:
        dict: mapping of original filename -> new filename (final target names)."""
    import uuid

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # compute initial targets
    target_counts = {}
    mapping = {}
    for f in files:
        if not f.startswith(ignore_prefix):
            continue
        name, ext = os.path.splitext(f)
        stripped = name[len(ignore_prefix):]
        prefix = stripped[:prefix_len]
        base_target = f"{ignore_prefix}{prefix}{ext}"

        # ensure uniqueness by using a counter per base_target
        count = target_counts.get(base_target, 0)
        if count == 0 and not os.path.exists(os.path.join(folder_path, base_target)):
            target = base_target
        else:
            # append suffix until unique
            i = count + 1
            while True:
                candidate = f"{ignore_prefix}{prefix}-{i}{ext}"
                if candidate not in mapping.values() and not os.path.exists(os.path.join(folder_path, candidate)):
                    target = candidate
                    break
                i += 1
        target_counts[base_target] = target_counts.get(base_target, 0) + 1
        mapping[f] = target

    if not mapping:
        print("No files to shorten (no files with given ignore_prefix).")
        return {}

    # dry-run: print changes and return
    if dry_run:
        print("Planned renames:")
        for src, dst in mapping.items():
            print(f"{src} -> {dst}")
        return mapping

    # perform two-step rename to temporary files to avoid collisions/overwrites
    temp_map = {}
    for src, dst in mapping.items():
        src_path = os.path.join(folder_path, src)
        if not os.path.exists(src_path):
            # skip if source disappeared in the meantime
            print(f"Source missing, skipping: {src}")
            continue
        temp_name = f".tmp_{uuid.uuid4().hex}_{src}"
        temp_path = os.path.join(folder_path, temp_name)
        os.replace(src_path, temp_path)
        temp_map[temp_path] = os.path.join(folder_path, dst)

    # move temps to final targets, ensuring final target uniqueness
    for temp_path, final_path in temp_map.items():
        final_dir, final_name = os.path.split(final_path)
        base, ext = os.path.splitext(final_name)
        candidate = final_name
        i = 1
        while os.path.exists(os.path.join(final_dir, candidate)):
            candidate = f"{base}-{i}{ext}"
            i += 1
        final_unique = os.path.join(final_dir, candidate)
        os.replace(temp_path, final_unique)

    print(f"Renamed {len(temp_map)} files in '{folder_path}'.")
    return mapping

#shorten_filenames(folder, prefix_len=5, ignore_prefix="train_", dry_run=False)


def shorten_metadata_keys(metadata_path, prefix_len=5, ignore_prefix="train_", out_path=None, backup=True):
    """Shorten keys in a metadata JSON file by keeping only `prefix_len` chars after `ignore_prefix`.

    - Only keys that start with `ignore_prefix` are changed.
    - Preserves file extension if present (e.g., .csv).
    - Ensures uniqueness by appending -1, -2... when collisions occur.
    - Creates a backup (.bak) when overwriting the original file.

    Returns a dict mapping old_key -> new_key for changed keys.
    """
    import json
    import shutil

    if out_path is None:
        out_path = metadata_path

    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    new_meta = {}
    mapping = {}
    used_targets = set()

    for old_key, value in meta.items():
        if not old_key.startswith(ignore_prefix):
            # leave unchanged
            new_meta[old_key] = value
            used_targets.add(old_key)
            continue

        name, ext = os.path.splitext(old_key)
        stripped = name[len(ignore_prefix):]
        prefix = stripped[:prefix_len]
        base_target = f"{ignore_prefix}{prefix}{ext}"

        target = base_target
        i = 1
        # if already used, append suffix
        while target in used_targets:
            # if base_target already has '-n' pattern, we still add -i
            target = f"{os.path.splitext(base_target)[0]}-{i}{ext}"
            i += 1

        new_meta[target] = value
        mapping[old_key] = target
        used_targets.add(target)

    # backup original
    if backup and out_path == metadata_path:
        bak_path = metadata_path + '.bak'
        shutil.copy2(metadata_path, bak_path)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_meta, f, indent=2)

    print(f"Converted {len(mapping)} metadata keys; wrote updated metadata to {out_path}")
    return mapping

shorten_metadata_keys(r'C:\Users\knutj\Code\BG_forecasting\standardized_datasets\Tidepool_HCL150\test_metadata.json', prefix_len=5, ignore_prefix='test_', out_path=None, backup=False)
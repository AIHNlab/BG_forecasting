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

shorten_filenames(folder, prefix_len=5, ignore_prefix="train_", dry_run=False)
from pathlib import Path
import re

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(__file__).resolve().parent
FLATTENED_FILE = SOURCE_DIR / "flattened_code.txt"

def restore_directory(flattened_file, source_dir):
    flattened_file = Path(flattened_file).resolve()
    source_dir = Path(source_dir).resolve()

    if not flattened_file.exists():
        print(f"ERROR: Flattened file not found:")
        print(flattened_file)
        return

    content = flattened_file.read_text(encoding="utf-8", errors="ignore")

    pattern = re.compile(
        r"#{20,}\s*"
        r"#\s*FILE\s+\d+\s*"
        r"#\s*PATH:\s*(.*?)\s*"
        r"#{20,}\s*"
        r"(.*?)"
        r"(?=\n#{20,}\s*\n#\s*FILE\s+\d+|\Z)",
        re.DOTALL
    )

    matches = pattern.findall(content)

    if not matches:
        print("ERROR: No file sections were found.")
        return

    print("=" * 70)
    print("RESTORING PROJECT")
    print("=" * 70)
    print(f"Source directory : {source_dir}")
    print(f"Flattened file   : {flattened_file}")
    print(f"Files detected   : {len(matches)}")
    print("=" * 70)

    restored = 0
    failed = 0

    for relative_path, file_content in matches:
        relative_path = relative_path.strip()
        target_file = source_dir / relative_path

        try:
            target_file = target_file.resolve()
            if not str(target_file).startswith(str(source_dir.resolve())):
                print(f"SKIPPED unsafe path: {relative_path}")
                failed += 1
                continue

            target_file.parent.mkdir(parents=True, exist_ok=True)

            if file_content.startswith("\n"):
                file_content = file_content[1:]

            target_file.write_text(file_content, encoding="utf-8")
            print(f"[OK] {relative_path}")
            restored += 1
        except Exception as e:
            print(f"[ERROR] {relative_path} -> {e}")
            failed += 1

    print()
    print("=" * 70)
    print("RESTORE COMPLETED")
    print("=" * 70)
    print(f"Successfully restored : {restored}")
    print(f"Failed                : {failed}")
    print("=" * 70)

if __name__ == "__main__":
    restore_directory(FLATTENED_FILE, SOURCE_DIR)

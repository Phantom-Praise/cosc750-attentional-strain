"""Build an explicit source ZIP without dataset or local execution debris."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent


def source_files(root=ROOT):
    files = [root/".gitignore", root/".gitattributes"]
    for pattern in ["*.py", "*.md", "requirements*.txt"]:
        files.extend(root.glob(pattern))
    for directory in ["tests", "docs", "research", "third_party_licenses", "reference_results"]:
        files.extend(p for p in (root/directory).rglob("*") if p.is_file()
                     and p.suffix in {".py", ".md", ".txt", ".json", ".csv", ".tex", ".bib", ".svg", ".png"}
                     and "__pycache__" not in p.parts)
    result = sorted(set(p for p in files if p.is_file()), key=lambda p: str(p.relative_to(root)))
    for p in result:
        if p.is_symlink(): raise ValueError(f"Refusing a symlink in source package: {p}")
    return result


def build(output, root=ROOT):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = {str(p.relative_to(root)): p.read_bytes() for p in source_files(root)}
    manifest = {name: hashlib.sha256(payload).hexdigest() for name, payload in entries.items()}
    entries["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            info = ZipInfo("cosc750-stage3-code/"+name, date_time=(2026, 11, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    print(f"Created {output.resolve()} ({len(entries)-1} files, {output.stat().st_size:,} bytes)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"dist/cosc750-stage3-code.zip")
    build(parser.parse_args().output)

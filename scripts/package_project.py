"""Create a portable local project archive without virtual environments or raw archives."""
from pathlib import Path
import json
import zipfile

from src.utils import dump_json, sha256


def package():
    root = Path.cwd()
    files = []
    for dirname in ["src", "scripts", "tests", "configs", "docs", "experiments", "logs", "data/processed", "results/public", "results/collection", "results/synthetic"]:
        files.extend(p for p in (root / dirname).rglob("*") if p.is_file() and "__pycache__" not in p.parts and "smoke" not in p.parts and p.suffix != ".pyc")
    for name in ["README.md", ".gitignore", "requirements.txt", "requirements.lock.txt", "pyproject.toml", "run_all.ps1"]:
        files.append(root / name)
    files.extend(root / "output" / name for name in ["Project_Report.pdf", "Project_Report.md", "report_content.json", "evidence_validation.json"])
    files = sorted(set(files))
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = dict(project="Campaign Shift Aware Social Bot Detection", date="2026-10-19", files=[dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha256(p)) for p in files],
        excluded=[".venv", "data/raw", "temporary rendering files", "development smoke-study outputs"],
        evidence_status=json.loads((root / "output/evidence_validation.json").read_text())["status"],
        research_boundary="Executed supported pilots; verified-campaign and real chronological validation remain outstanding")
    dump_json(root / "output/manifest.json", manifest)
    files.append(root / "output/manifest.json")
    archive = root / "output/Campaign_Shift_Bot_Detection_Project.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, arcname="Campaign_Shift_Bot_Detection/" + p.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as z:
        failed = z.testzip()
        if failed:
            raise ValueError("Archive CRC failure: " + failed)
        assert not any("/.venv/" in p or "/data/raw/" in p for p in z.namelist())
    dump_json(root / "output/package_checksum.json", dict(archive=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size, files=len(files)))
    print(archive, "MB", round(archive.stat().st_size/2**20, 2), "files", len(files))


if __name__ == "__main__":
    package()

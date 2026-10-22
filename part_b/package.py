"""Package code, measured results, documents and fitted model, without raw data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    required = ['part_b/deliverables/part_b_report.pdf',
                'part_b/deliverables/part_b_slides.pptx',
                'part_b/outputs/models/off_market_model.joblib',
                'submissions/submission_off_market.csv']
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError('Finish pipeline/documents before packaging: ' + name)
    files = [ROOT / name for name in ['README.md', '.gitignore', '.gitattributes', 'PART_B_START_HERE.md', 'data/README.md']
             if (ROOT / name).is_file()]
    for base in ['part_b', 'submissions']:
        files.extend(p for p in (ROOT / base).rglob('*') if p.is_file()
                     and not any(part in ['__pycache__', '.pytest_cache', 'cache'] for part in p.relative_to(ROOT).parts)
                     and p.suffix not in ['.pyc', '.zip'])
    files = sorted(set(files), key=lambda p: str(p.relative_to(ROOT)))
    entries = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    archive = ROOT / 'Part_B_Off_Market_Complete.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT))
        z.writestr('PART_B_MANIFEST.json', json.dumps(dict(files=entries,
            note='Raw input data, dependency environments and intermediate fold caches excluded.'), indent=2) + '\n')
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError('Corrupt ZIP member: ' + bad)
    print(f'Created and CRC-checked {archive} ({archive.stat().st_size:,} bytes; {len(files)+1} files)')


if __name__ == '__main__':
    main()

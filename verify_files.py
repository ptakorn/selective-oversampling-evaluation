"""Verify the prepared snapshot before modifying or regenerating its files."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;manifest=json.loads((R/'FILE_SHA256.json').read_text())
missing=[p for p in manifest if not (R/p).is_file()]
changed=[p for p,h in manifest.items() if (R/p).is_file() and hashlib.sha256((R/p).read_bytes()).hexdigest()!=h]
assert not missing and not changed,{'missing':missing,'changed':changed}
print(f'PASS: {len(manifest)} recorded files match SHA256 hashes. Regenerated/edited outputs may intentionally differ from this snapshot.')

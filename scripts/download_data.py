"""Download and verify the public Kaggle Abalone CSV used by the notebooks."""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://www.kaggle.com/api/v1/datasets/download/rodolfomendes/abalone-dataset"
SHA256 = "6fda3c4b0c0cff9bd5f7a46434509179069ed620c8fc7090d872a08d2c1a729a"


def download(output: Path) -> None:
    if output.exists():
        content = output.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            archive = response.read()
        with ZipFile(io.BytesIO(archive)) as zipped:
            names = [
                name
                for name in zipped.namelist()
                if Path(name).name.lower() == "abalone.csv"
            ]
            if len(names) != 1:
                raise ValueError("Download does not contain exactly one abalone.csv.")
            content = zipped.read(names[0])
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError(
            "Dataset checksum differs from the tested dataset; no file was overwritten."
        )
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
    print(f"Verified dataset: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/abalone.csv"))
    download(parser.parse_args().output)

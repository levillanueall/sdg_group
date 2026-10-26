import csv
from pathlib import Path


class CSVReader:

    def __init__(self, file_path: str, delimiter: str = ";"):
        self.file_path = Path(file_path)
        self.delimiter = delimiter

    def read_headers(self) -> list[str]:
        with self.file_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            reader = csv.reader(file, delimiter=self.delimiter)
            return next(reader)

    def read_rows(self):
        with self.file_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            reader = csv.DictReader(file, delimiter=self.delimiter)

            for row in reader:
                yield row
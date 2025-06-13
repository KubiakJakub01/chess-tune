import argparse
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from tqdm import tqdm

from ..utils import log_error, log_info, log_warning


def parse_args(raw_args: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Download data files from the Lichess public database.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--dates',
        '-d',
        nargs='+',
        metavar='YYYY-MM',
        help='One or more months to download (e.g. 2013-01 2013-02).',
    )
    group.add_argument(
        '--urls',
        '-u',
        nargs='+',
        metavar='URL',
        help='Download files from explicit URLs. Overrides --source and --dates.',
    )

    parser.add_argument(
        '--source',
        '-s',
        default='standard',
        choices=sorted(DATA_SOURCES),
        help='Predefined data source to download from.',
    )
    parser.add_argument(
        '--output',
        '-o',
        default='data/downloads',
        type=Path,
        help='Directory where downloaded files will be stored.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Re-download files even if they already exist locally.',
    )
    parser.add_argument(
        '--list-sources',
        action='store_true',
        help='Print available data sources and exit.',
    )

    parsed = parser.parse_args(args=raw_args)

    if parsed.list_sources:
        for ds in DATA_SOURCES.values():
            log_info(f'{ds.name:12s} → {ds.base_url}\n  {ds.description}\n')
        parser.exit()

    return parsed


@dataclass(frozen=True, slots=True)
class DataSource:
    """Represents a downloadable data source.

    Attributes
    ----------
    name:
        Short identifier used on the command line.
    base_url:
        Base URL without trailing slash (e.g. ``https://database.lichess.org/standard``).
    file_pattern:
        Python-format string used to build the file name from the provided
        *year* and *month* parameters.  Example::

            'lichess_db_standard_rated_{year}-{month}.pgn.zst'

        Both placeholders must be present in the pattern.
    description:
        Human-readable description shown when listing available sources.
    """

    name: str
    base_url: str
    file_pattern: str
    description: str = ''

    def build_url(self, year_month: str) -> str:
        """Return the full download URL for *year_month* (``YYYY-MM``)."""
        try:
            year, month = year_month.split('-', maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid date specifier '{year_month}'. Use the 'YYYY-MM' format, e.g. '2013-01'."
            ) from exc

        filename = self.file_pattern.format(year=year, month=month)
        return f'{self.base_url}/{filename}'


DATA_SOURCES: dict[str, DataSource] = {
    'standard': DataSource(
        name='standard',
        base_url='https://database.lichess.org/standard',
        file_pattern='lichess_db_standard_rated_{year}-{month}.pgn.zst',
        description='Lichess Standard rated games (monthly PGN, compressed with zstd)',
    ),
}


def download_file(
    url: str, output_dir: Path, overwrite: bool = False, chunk_size: int = 1 << 16
) -> Path:
    """Download *url* into *output_dir*, returning the local path.

    A partial download is first written to ``<filename>.part`` and renamed upon
    successful completion to avoid clobbering existing files when the download
    fails (e.g. connectivity loss).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit('/', maxsplit=1)[-1]
    dest_path = output_dir / filename

    if dest_path.exists() and not overwrite:
        log_warning(f'[SKIP] {dest_path} already exists. Use --overwrite to re-download.')
        return dest_path

    tmp_path = dest_path.with_suffix(dest_path.suffix + '.part')

    try:
        with urlopen(url) as response:
            total = response.length or 0
            with tmp_path.open('wb') as fp:
                for data in tqdm(
                    iterable=iter(lambda: response.read(chunk_size), b''),
                    total=total // chunk_size,
                    unit='chunk',
                    unit_scale=False,
                    desc=filename,
                ):
                    fp.write(data)

    except (HTTPError, URLError) as exc:
        log_error(f'[ERROR] Could not download {url}: {exc}')
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(dest_path)
    log_info(f'[DONE] {dest_path}')
    return dest_path


def main(raw_args: list[str] | None = None) -> None:
    args = parse_args(raw_args)

    urls: list[str]
    if args.urls:
        urls = args.urls
    else:
        datasource: DataSource = DATA_SOURCES[args.source]
        urls = [datasource.build_url(spec) for spec in args.dates]

    for url in urls:
        try:
            download_file(url, args.output, overwrite=args.overwrite)
        except Exception:
            log_error(f'[FAIL] Download failed for {url}')
            continue


if __name__ == '__main__':
    main()

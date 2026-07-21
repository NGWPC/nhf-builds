import argparse
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import RequestException

BASE_URL = "https://cwms-data.usace.army.mil/cwms-data/"
GROUPS = "location/group?location-category-like=flow"
LOCATION_URL = "locations/{site}?office={office}"


def download_usace(max_retries: int = 3, output_file: Path = Path("./usace_gages.csv")) -> None:
    """Download USACE locations with flow type data.

    Parameters
    ----------
    max_retries : int, optional
        Max retries for HTTP requests, by default 3
    output_file : Path, optional
        Output CSV, by default Path("./usace_gages.csv")
    """
    # get list of location groups with flow category
    for attempt in range(max_retries):
        try:
            groups = requests.get(f"{BASE_URL}{GROUPS}", timeout=60)
            groups.raise_for_status()
            data = groups.json()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            else:
                raise e

    # pairs of office ID and location ID
    vals = [
        (i["office-id"], i["shared-ref-location-id"])
        for i in data
        if i.get("shared-ref-location-id", None) is not None
    ]

    # de-duplicate office-id, location ID pairings - there are often duplicates due to multiple types of flow datasets available
    unique = []
    for val in vals:
        if val not in unique:
            unique.append(val)

    # query the location URL for each location/office pairing and save relevant data (lat, lon, long-name when available)
    output = []
    failed = []
    for val in unique:
        for attempt in range(max_retries):
            try:
                loc_url = LOCATION_URL.format(site=val[1], office=val[0], timeout=60)
                r = requests.get(f"{BASE_URL}{loc_url}")
                r.raise_for_status()
                data = r.json()
                try:
                    output.append((val[1], val[0], data["latitude"], data["longitude"], data["long-name"]))
                # sometimes long-name is missing
                except KeyError:
                    output.append((val[1], val[0], data["latitude"], data["longitude"], None))
                break
            except RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Retrying {val[1]}")
                    continue
                else:
                    failed.append((val[1], str(e)))
                    print(f"Failed {val[1]} for {str(e)}")
                    continue
            except Exception as e:  # noqa: BLE001
                failed.append((val[1], str(e)))
                print(f"Failed {val[1]} for {str(e)}")
                continue

    df = pd.DataFrame.from_records(
        data=output, columns=["location", "office", "latitude", "longitude", "full-name"]
    )

    df.to_csv(output_file)
    print(f"Failed gages: {len(failed)}")


def crosswalk_usace() -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default="./usbr.gpkg",
        help="Output path for USBR data",
    )
    args = parser.parse_args()
    download_usace(output_file=args.output)

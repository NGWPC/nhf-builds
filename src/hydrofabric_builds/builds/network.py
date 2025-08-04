import pandas as pd
from pyarrow import fs
from pyiceberg.catalog import Catalog

from hydrofabric_builds.schemas.hydrofabric import HydrofabricDomains


def update_network_poi(
    catalog: Catalog, hf_domain: str, poi_path: str, s3: fs.S3FileSystem | None = None
) -> pd.DataFrame:
    """Update a Hydrofabric network table with new POIs from a POI table

    Parameters
    ----------
    catalog : Catalog
        Pyicbeger catalog to read from
    hf_domain : str
        Hydrofabric Domain to use
    poi_path : _type_
        Path to new POI parquet
    s3 : fs.S3FileSystem | None, optional
        An s3 filesystem if reading and writing to s3, by default None

    Returns
    -------
    pd.DataFrame
        Updated network table with new POIs
    """
    print("Updating network table")
    df_network = catalog.load_table(f"{HydrofabricDomains[hf_domain].value}.network").scan().to_pandas()

    df_poi = pd.read_parquet(poi_path, filesystem=s3) if s3 else pd.read_parquet(poi_path)

    # merge on nex_id only
    df_network_merge = df_network.merge(
        df_poi[["poi_id", "nex_id"]], how="left", left_on=["toid"], right_on=["nex_id"]
    )

    # replace old POI with new POI
    df_network_merge["poi_id_x"] = df_network_merge["poi_id_y"]
    df_network_merge = df_network_merge.drop(columns=["poi_id_y", "nex_id"]).rename(
        columns={"poi_id_x": "poi_id"}
    )

    return df_network_merge

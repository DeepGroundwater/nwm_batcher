import warnings

import fsspec
import xarray as xr
from datetime import datetime, timedelta
from virtualizarr import open_virtual_dataset
from dask.distributed import LocalCluster

from nwm_batcher.data_types import validate_configuration

warnings.filterwarnings("ignore", category=UserWarning)

def _process(url: str, so: dict[str, str]):
    vds = open_virtual_dataset(
            url, 
            drop_variables="crs",
            # loadable_variables=["streamflow"],
            indexes={}, 
            reader_options={"storage_options": so}
        )
    return vds


def read(
    date: str,
    forecast_type: str,
    initial_time: str = "t00z",
    variable: str = "channel_rt",
    client_settings: dict[str, int | str] = {
        "n_workers":9,
        "memory_limit":"2GiB",
    },
):
    fs = fsspec.filesystem("s3", anon=True)
    
    cluster= LocalCluster(**client_settings)  
    client = cluster.get_client()
    
    assert validate_configuration(forecast_type)

    file_pattern = f"s3://noaa-nwm-pds/nwm.{date}/{forecast_type}/nwm.{initial_time}.{forecast_type}.{variable}.*.nc"
    noaa_files = fs.glob(file_pattern)
    noaa_files = sorted(["s3://" + f for f in noaa_files])
    
    so = dict(anon=True, default_fill_cache=False, default_cache_type="none")

    futures = []
    for url in noaa_files:  
        future = client.submit(_process, url, so)
        futures.append(future)

    virtual_datasets = client.gather(futures)  

    virtual_ds = xr.combine_nested(
        virtual_datasets, 
        coords="minimal", 
        compat='override', 
        concat_dim=['time']
    )
    virtual_ds = virtual_ds.chunk({'time':1}, chunked_array_type="cubed")
    return virtual_ds

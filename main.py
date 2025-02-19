import warnings

import fsspec
import xarray as xr
import dask
import coiled
from datetime import datetime, timedelta
from virtualizarr import open_virtual_dataset
import icechunk
from dask.distributed import LocalCluster, Client

warnings.filterwarnings("ignore", category=UserWarning)

def process(url: str, so: dict[str, str]):
    vds = open_virtual_dataset(
            url, 
            drop_variables="crs",
            loadable_variables=["streamflow"],
            indexes={}, 
            reader_options={"storage_options": so}
        )
    return vds


def main():
    fs = fsspec.filesystem("s3", anon=True)

    date = "20250216"
    forecast_type = "short_range"
    
    cluster= LocalCluster()  
    client = cluster.get_client()

    # Get the first day's pattern
    file_pattern = f"s3://noaa-nwm-pds/nwm.{date}/{forecast_type}/nwm.t00z.{forecast_type}.*.*.nc"
    noaa_files = fs.glob(file_pattern)
    noaa_files = sorted(["s3://" + f for f in noaa_files])
    
    so = dict(anon=True, default_fill_cache=False, default_cache_type="none")

    futures = []
    for url in noaa_files:  
        future = client.submit(process, url, so)
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

if __name__ == "__main__":
    main()

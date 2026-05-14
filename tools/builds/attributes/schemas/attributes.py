from enum import Enum


class GroundWaterProjectionCONUS(Enum):
    """CRS, origin (top, left corner), and size of CONUS NWM grids for groundwater; source:  Fulldom_CONUS_FullRouting.nc"""

    PROJ4 = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40.0000076293945 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"
    X_ORIGIN = -2303874.17655
    Y_ORIGIN = -1919874.66329
    WIDTH = 18432
    HEIGHT = 15360
    DX = 250
    DY = 250


class GroundWaterProjectionAK(Enum):
    """CRS, origin (top, left corner), and size of NWM grids for groundwater; source:  Fulldom_AK_FullRouting.nc"""

    PROJ4 = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-135 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs"
    X_ORIGIN = -1130764.7202253528
    Y_ORIGIN = -3163389.53353531
    WIDTH = 3516
    HEIGHT = 1816
    DX = 250
    DY = 250


class GroundWaterProjectionHI(Enum):
    """CRS, origin (top, left corner), and size of NWM grids for groundwater; source:  Fulldom_HI_FullRouting.nc"""

    PROJ4 = "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=10.0 +lat_2=30.0 +lat_0=20.6 +lon_0=-157.42 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext +no_defs"
    X_ORIGIN = -294950.07097397465
    Y_ORIGIN = -194949.36969098
    WIDTH = 5900
    HEIGHT = 3900
    DX = 100
    DY = 100


class GroundWaterProjectionPRVI(Enum):
    """CRS, origin (top, left corner), and size of NWM grids for groundwater; source:  Fulldom_PRVI_FullRouting.nc"""

    PROJ4 = "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=18.1 +lat_2=18.1 +lat_0=18.1 +lon_0=-65.91 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs"
    X_ORIGIN = -149949.83
    Y_ORIGIN = -54948.968
    WIDTH = 3000
    HEIGHT = 1100
    DX = 100
    DY = 100


class NWMProjectionCONUS(Enum):
    """Projection for NWM soilproperties for CONUS"""

    PROJ4 = "+proj=lcc +lat_0=40.0000076293945 +lon_0=-97 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs"
    XMIN = -2304000
    XMAX = 2304000
    YMIN = -1920001
    YMAX = 1919999
    DX = 1000
    DY = 1000


class NWMProjectionAK(Enum):
    """Projection for NWM soilproperties for AK"""

    PROJ4 = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-135"
    XMIN = -1133617.2
    XMAX = -254617.2
    YMIN = -3175222.3
    YMAX = -2716222.3
    DX = 1000
    DY = 1000


class NWMProjectionHI(Enum):
    """Projection for NWM soilproperties for HI"""

    PROJ4 = "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=10.0 +lat_2=30.0 +lat_0=20.6 +lon_0=-157.42 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs"
    XMIN = -295000.4746870845
    XMAX = 294999.5253129155
    YMIN = -194999.9185492387
    YMAX = 195000.0814507613
    DX = 1000
    DY = 1000


class NWMProjectionPRVI(Enum):
    """Projection for NWM soilproperties for PRVI"""

    PROJ4 = "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=10.0 +lat_2=30.0 +lat_0=20.6 +lon_0=-157.42 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs"
    XMIN = -295000.4746870845
    XMAX = 294999.5253129155
    YMIN = -194999.9185492387
    YMAX = 195000.0814507613
    DX = 1000
    DY = 1000


class HydrofabricCRS(Enum):
    """The domains used when querying the hydrofabric

    Attributes
    ----------
    AK : str
        Alaska
    CONUS : str
        Conterminous United States
    GL : str
        The US Great Lakes
    HI : str
        Hawai'i
    PRVI : str
        Puerto Rico, US Virgin Islands
    """

    AK = 3338
    CONUS = 5070
    HI = 32604
    PRVI = 6566

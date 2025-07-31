import geopandas as gpd

mylakes_file = "lakes.gpkg"
hflakes_file = "conus_nextgen.gpkg"
output_file = "diff.csv"

mylakes = gpd.read_file(mylakes_file, layer="lakes")
hflakes = gpd.read_file(hflakes_file, layer="lakes")

mylakes = mylakes.drop("feature_id", axis=1)

hflakes = hflakes.sort_values("lake_id")
mylakes = mylakes.sort_values("lake_id")

hflakes = hflakes.set_index("lake_id")
mylakes = mylakes.set_index("lake_id")

mylakes_index = mylakes.index.to_list()
hflakes_index = hflakes.index.to_list()

if len(mylakes) != len(hflakes):
    lakeid = [item for item in hflakes_index if item not in mylakes_index]
    if lakeid:
        print(f"{lakeid} in hf lakes but not in my lakes")
        hflakes = hflakes.drop(lakeid)

    lakeid = [item for item in mylakes_index if item not in hflakes_index]
    if lakeid:
        print(f"{lakeid} in my lakes but not in hf lakes")
        mylakes = mylakes.drop(lakeid)

diff = mylakes.compare(hflakes)

diff.to_csv(output_file)

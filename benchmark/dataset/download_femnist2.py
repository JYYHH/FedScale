import flbenchmark.datasets

flbd = flbenchmark.datasets.FLBDatasets('../data')

print("Downloading femnist2 benchmark Data...")

my_dataset = flbd.leafDatasets('femnist')

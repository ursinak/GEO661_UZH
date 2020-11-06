import pyspark
import sys
import pyspark.sql.functions as sql

print(sys.version)
print(sys.path)

from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window

import pandas as pd


# set folder structure/ data to import
folder_id = '22740'
data_root_path = 'inter-data'
raw_fn = folder_id + '.csv'

# schema of cvs to be imported
schema = StructType([
                    StructField('vehicle', StringType()),
                     StructField('engineStatus', IntegerType()),
                     StructField('driverEvent', StringType()),
                     StructField('localDate', TimestampType()),
                     StructField('longitude', DoubleType()),
                     StructField('latitude', DoubleType()),
                     StructField('traj_id', IntegerType()),
                     StructField('wp_seq', IntegerType()),
                     StructField('x', DoubleType()),
                     StructField('y', DoubleType()),
                     StructField('zorder', LongType())])


df = spark.read.option("header","false").schema(schema)    .csv(os.path.join(data_root_path, raw_fn))    


# origin
w = Window.partitionBy('traj_id', 'vehicle')
trip_origin = df.withColumn('min_wp_seq', F.min('wp_seq').over(w)).where(F.col('wp_seq') == F.col('min_wp_seq')).drop('min_wp_seq')

# destination
trip_destination = df.withColumn('max_wp_seq', F.max('wp_seq').over(w)).where(F.col('wp_seq') == F.col('max_wp_seq')).drop('max_wp_seq')



# Convert Z-Order (int) to binary code

def z_to_z_str(z):
    bin_z = bin(z)[2:]
    #Fill in the missing 0s. For instance, if a node is 0->0->1, 
    #the binary only returns 1 digit: 1, but the actually represent is 001
    bin_z = '0' * (48 - len(bin_z)) + bin_z
    return bin_z

# rename columns - origin
trip_origin = trip_origin.withColumnRenamed('vehicle', 'vehicle_O')    .withColumnRenamed('engineStatus', 'O_engineStatus')    .withColumnRenamed('localDate', 'O_localDate')    .withColumnRenamed('longitude', 'O_longitude')    .withColumnRenamed('latitude', 'O_latitude')    .withColumnRenamed('wp_seq', 'O_wp_seq')    .withColumnRenamed('x', 'O_x')    .withColumnRenamed('y', 'O_y')    .withColumnRenamed('zorder', 'O_zorder')    

# destination
trip_destination = trip_destination.withColumnRenamed('vehicle', 'D_vehicle')    .withColumnRenamed('engineStatus', 'D_engineStatus')    .withColumnRenamed('localDate', 'D_localDate')    .withColumnRenamed('longitude', 'D_longitude')    .withColumnRenamed('latitude', 'D_latitude')    .withColumnRenamed('wp_seq', 'D_wp_seq')    .withColumnRenamed('x', 'D_x')    .withColumnRenamed('y', 'D_y')    .withColumnRenamed('zorder', 'D_zorder')

# join origin and destination df
trip_od = trip_origin.join(trip_destination, trip_origin.traj_id == trip_destination.traj_id)
trip_od = trip_od.where(trip_od.D_wp_seq != trip_od.O_wp_seq)

# get number of trips in origin-destination df
trip_od_recs = trip_od.collect()
print(len(trip_od_recs))

# get vehicle numbers of current file
vehicles = [row['vehicle_O'] for row in trip_od_recs]
veh = []
for v in vehicles:
    if v not in veh:
        veh.append(v)


# get pandas dataframe of trip_od
trip_od_recs_pd = pd.DataFrame.from_records(trip_od_recs)
trip_od_recs_pd.columns = trip_od_recs[0].__fields__


# compute variance of the flow weights
import numpy as np
trip_od_veh = trip_od_recs_pd.loc[trip_od_recs_pd['vehicle_O'] == '160_138670'] # only show OD for specific vehicle

o_z = []
d_z = []
for row in trip_od_veh['O_zorder']: # get all Z-Orders for the origins
     o_z.append(row)

for row in trip_od_veh['D_zorder']: # get all Z-Orders for the destinations
    d_z.append(row)

# convert z-order to binary and save as string
o_z_str = [z_to_z_str(z) for z in o_z]
d_z_str = [z_to_z_str(z) for z in d_z]

od_z_str = [r for r in zip(o_z_str, d_z_str)]

# get max and second max value for each level
od_level = []
cnt = Counter()
variance_level = []

for level in range(1, 25): # get the variance for each level
    cnt = Counter()
    x_od_zorder = [(r[0][:2 * level], r[1][:2 * level]) for r in od_z_str]
    for nr in x_od_zorder:
        cnt[nr] += 1
    x_od_zorder_set = set(cnt)
    od_level.append([cnt])
    
    mostcommon = cnt.most_common()
    vrnc = np.var([count for key, count in mostcommon])
    variance_level.append(vrnc)


# compute the degree of primacy
trip_od_veh = trip_od_recs_pd.loc[trip_od_recs_pd['vehicle_O'] == '7580_98860']

o_z = []
d_z = []
for row in trip_od_veh['O_zorder']:
     o_z.append(row)

for row in trip_od_veh['D_zorder']:
    d_z.append(row)

# convert z-order to binary and save as string
o_z_str = [z_to_z_str(z) for z in o_z]
d_z_str = [z_to_z_str(z) for z in d_z]

od_z_str = [r for r in zip(o_z_str, d_z_str)]

# get max and second max value for each level
od_level = []
cnt = Counter()
maxvals = []
dop = []

for level in range(1, 25): #until 25
    cnt = Counter()
    x_od_zorder = [(r[0][:2 * level], r[1][:2 * level]) for r in od_z_str]
    for nr in x_od_zorder:
        cnt[nr] += 1
    x_od_zorder_set = set(cnt)
    od_level.append([cnt])
    dopMax = cnt.most_common()[:][0][1]
    if len(cnt.most_common(2)) < 2:
        maxvals.append([dopMax])
    else:
        dopSec = cnt.most_common()[:2][1][1]
        maxvals.append([dopMax, dopSec])


# calculate dop 
for val in maxvals:
    if len(val) < 2:
        m = 0
    else:
        m = val[0]/val[1]
    dop.append(m)


# calculate the number of unique z-order values
o_z_str_unique = set(o_z_str)
d_z_set_unique = set(d_z_str)
print(len(o_z_str), len(d_z_str))
print(len(o_z_str_unique), len(d_z_set_unique), len(o_z_str_unique.union(d_z_set_unique)))

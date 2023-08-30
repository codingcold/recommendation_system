# Method Description:
#Parse the input arguments, which are the paths to the folder containing the data, the input file, and the output file.
#Load the training data from the input file, which is assumed to be in CSV format, and convert it to an RDD of tuples. Each tuple consists of a pair (user_id, business_id) and a float representing the rating given by the user to the business.
#Load the user, business, photo, tip, and checkin data from the JSON files and convert them to RDDs of dictionaries. These RDDs will be used to extract features for each user and business.
#Compute the features for each user and business and store them in dictionaries, where the keys are the user or business IDs and the values are lists of feature values.
#Load the validation data from the validation CSV file and convert it to an RDD of tuples. Each tuple consists of a triple (user_id, business_id, rating).
#Collect the training data and validation data as lists of tuples, where each tuple consists of a list of features and a rating.
#Train an XGBoost model on the training data using the parameters specified in the code.
#Use the trained model to make predictions on the validation data.
#Write the predictions to the output file in CSV format, where each row contains a triple (user_id, business_id, prediction).

# Error Distribution:
# >=0 and <1: 102156
# >=1 and <2: 32944
# >=2 and <3: 6151
# >=3 and <4: 792
# >=4: 1

# Duration: 311.5854136943817
# RMSE: 0.9231368038873942


import time
import sys
from pyspark import SparkContext
import xgboost as xgb
from operator import add
import json
import numpy as np

folder_path = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]

v = "yelp_val.csv"
u = "user.json"
b = "business.json"
p = "photo.json"
c = "checkin.json"
t = "tip.json"

val_path = folder_path + v
user_data = folder_path + u
bs_data = folder_path + b
photo_data = folder_path + p
check_in = folder_path + c
tip_data = folder_path + t

sc = SparkContext.getOrCreate()
sc.setLogLevel('Error')

content = sc.textFile(input_path).map(lambda l: l.split(",")).filter(lambda l: l[0] != 'user_id')
train_data = content.map(lambda s: ((s[0], s[1]), float(s[2])))

u_data = sc.textFile(user_data).map(lambda r: json.loads(r))
bs = sc.textFile(bs_data).map(lambda r: json.loads(r))
photo = sc.textFile(photo_data).map(lambda r: json.loads(r))
tip = sc.textFile(tip_data).map(lambda r: json.loads(r))
checkin = sc.textFile(check_in).map(lambda r: json.loads(r))


u_feature = u_data.map(lambda x: (x['user_id'], (
float(x['review_count']), float(x['average_stars']), int(x['useful']) + int(x['funny']) + int(x['cool']),
int(x['compliment_hot']) + int(x['compliment_more']) + int(x['compliment_profile']) + int(x['compliment_cute']) + int(
    x['compliment_list']) + int(x['compliment_note']) + int(x['compliment_plain']) + int(x['compliment_cool']) + int(
    x['compliment_funny']) + int(x['compliment_writer']) + int(x['compliment_photos']), int(x['fans']),
int(len(x["elite"]) if x["elite"] is not None else "0")))).collectAsMap()

b_feature = bs.map(lambda x: (x['business_id'], (
float(x['review_count']), float(x['stars']), (float(x["longitude"]) + 180) / 360 if x["longitude"] is not None else 0.5,
(float(x["latitude"]) + 90) / 180 if x["latitude"] is not None else 0.5, int(x['is_open']),
int(len(x["categories"]) if x["categories"] is not None else "0")))).collectAsMap()

p_feature = photo.map(lambda x: (x['business_id'], 1)).reduceByKey(add).collectAsMap()

t_feature = tip.map(lambda x: ((x['business_id'], x['user_id']), x["likes"])).reduceByKey(add).collectAsMap()



US_FEAT_LENGTH = 6
BS_FEAT_LENGTH = 6

def collection(usr, bsness):
    if usr not in u_feature or bsness not in b_feature:
        return None

    us_feature = u_feature.get(usr, [np.nan] * US_FEAT_LENGTH)
    bs_feature = b_feature.get(bsness, [np.nan] * BS_FEAT_LENGTH)
    pv_feature = p_feature.get(bsness, np.nan)
    tv_feature = t_feature.get((bsness, usr), np.nan)

    feat_ar = [*us_feature, *bs_feature, pv_feature, tv_feature, pv_feature]

    return feat_ar

val_data0 = sc.textFile(val_path).map(lambda l: l.split(",")).filter(lambda l: l[0] != 'user_id')
val_ub = val_data0.map(lambda x: (x[0], x[1], float(x[2])))
val_data = val_ub.collect()

start = time.time()

train_x = train_data.map(lambda x: collection(x[0][0], x[0][1])).collect()
train_y = train_data.map(lambda x: x[1]).collect()

test_x = val_ub.map(lambda k: collection(k[0], k[1])).collect()

model_try = xgb.XGBRegressor(objective='reg:linear', learning_rate=0.1, max_depth=5, n_estimators=700, reg_lambda=1.5, n_jobs=-1)

model_try.fit(train_x, train_y)
prediction = model_try.predict(test_x)


try:
    f = open(output_path, "w")
    f.write("user_id, business_id, prediction\n")
    for j in range(len(val_data)):
        f.write(val_data[j][0] + "," + val_data[j][1] + "," + str(prediction[j]) + "\n")
    f.close()
except Exception as e:
    print("Error writing output file:", e)

end = time.time()
print('Duration:', end - start)
import warnings
import numpy as np
import pandas as pd
from matplotlib import patches
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import concurrent.futures
import time
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
warnings.filterwarnings("ignore")

dataset_train=pd.read_csv('kdd_train.csv')

dataset_test=pd.read_csv('kdd_test.csv')


col_names = ["duration","protocol_type","service","flag","src_bytes",
     "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
     "logged_in","num_compromised","root_shell","su_attempted","num_root",
     "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
     "is_host_login","is_guest_login","count","srv_count","serror_rate",
     "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
     "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
     "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
     "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
     "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


print("Shape of Training Dataset:", dataset_train.shape)
print("Shape of Testing Dataset:", dataset_test.shape)

# # Assigning attribute name to dataset
dataset_train = pd.read_csv("kdd_train.csv", header=None, names = col_names)
dataset_test = pd.read_csv("kdd_test.csv", header=None, names = col_names)

# #label distribution of Training set and testing set
print('Label distribution Training set:')
print(dataset_train['label'].value_counts())
print()
print('Label distribution Test set:')
print(dataset_test['label'].value_counts())

# # colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# # explore categorical features
print('Training set:')
for col_name in dataset_train.columns:
    if dataset_train[col_name].dtypes == 'object' :
         unique_cat = len(dataset_train[col_name].unique())
         print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

# #see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
# print()
print('Distribution of categories in service:')
print(dataset_train['service'].value_counts().sort_values(ascending=False).head())


# # Test set
print('Test set:')
for col_name in dataset_test.columns:
     if dataset_test[col_name].dtypes == 'object' :
         unique_cat = len(dataset_test[col_name].unique())
         print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

#categorical_columns=['protocol_type', 'service', 'flag']
# # insert code to get a list of categorical columns into a variable, categorical_columns
categorical_columns=['protocol_type', 'service', 'flag']
#  # Get the categorical values into a 2D numpy array
dataset_train_categorical_values = dataset_train[categorical_columns]
dataset_test_categorical_values = dataset_test[categorical_columns]
dataset_train_categorical_values.head()

# # protocol type
unique_protocol=sorted(dataset_train.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
# # service
unique_service=sorted(dataset_train.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
# # flag
unique_flag=sorted(dataset_train.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
# # put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)

# #do same for test set
unique_service_test=sorted(dataset_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2

# #Transform categorical features into numbers using LabelEncoder()
dataset_train_categorical_values_enc=dataset_train_categorical_values.apply(LabelEncoder().fit_transform)
print(dataset_train_categorical_values_enc.head())
# # test set
dataset_test_categorical_values_enc=dataset_test_categorical_values.apply(LabelEncoder().fit_transform)


# #One-Hot-Encoding¶
enc = OneHotEncoder()
dataset_train_categorical_values_encenc = enc.fit_transform(dataset_train_categorical_values_enc)
dataset_train_cat_data = pd.DataFrame(dataset_train_categorical_values_encenc.toarray(),columns=dumcols)
# # test set
dataset_test_categorical_values_encenc = enc.fit_transform(dataset_test_categorical_values_enc)
dataset_test_cat_data = pd.DataFrame(dataset_test_categorical_values_encenc.toarray(),columns=testdumcols)

dataset_train_cat_data.head()


trainservice=dataset_train['service'].tolist()
testservice= dataset_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
print(difference)

for col in difference:
     dataset_test_cat_data[col] = 0

print(dataset_test_cat_data.shape)

# #Join encoded categorical dataframe with the non-categorical dataframe
newdf=dataset_train.join(dataset_train_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
# # test data
newdf_test=dataset_test.join(dataset_test_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
print(newdf.shape)
print(newdf_test.shape)

# # take label column
labeldf=newdf['label']
labeldf_test=newdf_test['label']
# # change the label column
newlabeldf=labeldf.replace({ 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                            'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                            ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                            'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                            'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                            ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                            'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})

# # put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test

y_train= newdf['label']
y_test= newdf_test['label']
import csv
with open('newdataset/labeltrain.csv', 'a', newline='') as csvfile:
     spamwriter = csv.writer(csvfile, delimiter=',')
     #for row in rows:
     spamwriter.writerows(map(lambda x: [x], y_train))
with open('newdataset/labeltest.csv', 'a', newline='') as csvfile:
     spamwriter = csv.writer(csvfile, delimiter=',')
     #for row in rows:
     spamwriter.writerows(map(lambda x: [x], y_test))
X_train=newdf.drop('label',axis='columns')
X_train = X_train[1:] #take the data less the header row
for i in range(len(X_train)):
    with open('newdataset/train.csv', 'a', newline='') as csvfile:
     spamwriter = csv.writer(csvfile, delimiter=',')
     #for row in rows:
     spamwriter.writerow(X_train.iloc[i])
new_header = X_train.iloc[0] 
X_train.columns = new_header

X_test=newdf_test.drop('label',axis='columns')
X_test = X_test[1:] #take the data less the header row
for i in range(len(X_test)):
   with open('newdataset/test.csv', 'a', newline='') as csvfile:
     spamwriter = csv.writer(csvfile, delimiter=',')
     #for row in rows:
     spamwriter.writerow(X_test.iloc[i])

X_test.columns = new_header
new_header = X_test.iloc[0] 
X_test.columns = new_header

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
#from minisom import minisom
data= pd.read_csv('Dataset\KDDTrain+.csv')
# X is all columns except last one class
# Y is Class - we use it to verify after clustering (SOM) is done 
#X = data.iloc[::-2].values
#Y = data.iloc[::42].values
dos_attacks=["snmpgetattack","back","land","neptune","smurf","teardrop","pod","apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpguess","worm","httptunnel","named","xlock","xsnoop","sendmail","ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit","xterm","ps"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]
classes=["Normal","Dos","R2L","U2R","Probe"]
#Helper function to label samples to 5 classes
row= data.iloc[:, 42].values
def label_attack (row):
    if row in dos_attacks:
        return classes[1]
    if row in r2l_attacks:
        return classes[2]
    if row in u2r_attacks:
        return classes[3]
    if row in probe_attacks:
        return classes[4]
    return classes[0]
print(label_attack)
data["Class"]=data.apply(label_attack,axis=1)
 #For One Hot Encoding all categorical data
label_encoder_1 = preprocessing.LabelEncoder()
label_encoder_2 = preprocessing.LabelEncoder()
label_encoder_3 = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder(categories= 'auto')
data[42] = ''
data = data.values
data[:, 1] = label_encoder_1.fit_transform(data[:, 1])
data[:, 2] = label_encoder_2.fit_transform(data[:, 2])        
data[:, 3] = label_encoder_3.fit_transform(data[:, 3])
data_features = one_hot_encoder.fit_transform(data[:, :-2]).toarray() 
print(data_features)        
transform = ColumnTransformer([( 'transform',OneHotEncoder(categories = "auto"), [1,2,3])],remainder='passthrough')
data = transform.fit_transform(data)
print(data.shape)
"""
def encodingLabels(Y,X,data):
   attackType  = {'normal': 'normal', 'neptune':'DoS', 'warezclient': 'R2L', 'ipsweep': 'Probe','back': 'DoS', 'smurf': 'DoS', 'rootkit': 'U2R','satan': 'Probe', 'guess_passwd': 'R2L','portsweep': 'Probe','teardrop': 'DoS','nmap': 'Probe','pod': 'DoS','ftp_write': 'R2L','multihop': 'R2L','buffer_overflow': 'U2R','imap': 'R2L','warezmaster': 'R2L','phf': 'R2L','land': 'DoS','loadmodule': 'U2R','spy': 'R2L','perl': 'U2R'} 
   attackEncodingCluster  = {'normal':0,'DoS':1,'Probe':2,'R2L':3, 'U2R':4}
   Y[:] = [attackType[item] for item in Y[:]]
   Y[:] = [attackEncodingCluster[item] for item in Y[:]]
   return data
from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range=(0,1))
#data = sc.fit_transform(data)
"""
"""
def scaleColumns(df, cols_to_scale):
    for key in attackEncodingCluster:
        data[data[:, 41] == key, 42] = attackType[key]
        df[key] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(dfTest[key])),columns=[key])
    return df
  
"""
#from minisom import Minisom
#som = minisom(x=11,y=11,input_len=124, sigma=1.0, learning_rate=0.5,decay_function=asymptotic_decay,
                 #neighborhood_function='gaussian', topology='rectangular',
                 #activation_distance='euclidean', random_seed= 10 )
#som.random_weights_init(X)
#som.train_random(data = X, num_iteration = 100)
#som.distance_map().shape

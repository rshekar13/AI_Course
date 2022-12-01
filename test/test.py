#import required modules
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')

import joblib
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import time

single_val_features = ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289',
                       'X290', 'X293', 'X297', 'X330', 'X347']

single_val_feat_from_bin = ['X15', 'X16', 'X17', 'X18', 'X21', 'X24', 'X26', 'X30', 'X33', 'X34', 'X36', 'X39', 'X40', 'X42', 'X53', 'X55', 'X59',
 'X60', 'X62', 'X65', 'X67', 'X74', 'X78', 'X83', 'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X94', 'X95', 'X97', 
 'X99', 'X102', 'X104', 'X105', 'X110', 'X112', 'X122', 'X123', 'X124', 'X125', 'X145', 'X153', 'X160', 'X165', 'X167', 
 'X169', 'X172', 'X173', 'X183', 'X184', 'X190', 'X192', 'X199', 'X200', 'X204', 'X205', 'X207', 'X210', 'X212', 'X213',
 'X214', 'X216', 'X217', 'X221', 'X227', 'X230', 'X236', 'X237', 'X239', 'X240', 'X242', 'X243', 'X245', 'X248', 'X249', 
 'X252', 'X253', 'X254', 'X257', 'X258', 'X259', 'X260', 'X262', 'X266', 'X267', 'X269', 'X270', 'X271', 'X274', 'X277', 
 'X278', 'X280', 'X281', 'X282', 'X288', 'X292', 'X295', 'X296', 'X298', 'X299', 'X307', 'X308', 'X309', 'X310', 'X312', 
 'X317', 'X318', 'X319', 'X320', 'X323', 'X325', 'X332', 'X335', 'X338', 'X339', 'X341', 'X344', 'X353', 'X357', 'X364', 
 'X365', 'X366', 'X369', 'X370', 'X372', 'X379', 'X380', 'X382', 'X383', 'X384', 'X385'] 


cat_list = ['X0', 'X1', 'X2', 'X3', 'X5', 'X6', 'X8']


kbest = [  0,   1,   4,   7,   8,   9,  11,  14,  15,  17,  19,  22,  25,
        26,  30,  34,  35,  37,  38,  39,  42,  43,  44,  45,  52,  56,
        62,  71,  74,  75,  76,  80,  85,  87,  89,  99, 102, 103, 105,
       106, 110, 111, 112, 116, 118, 119, 121, 168, 169, 172]


def preprocessing(test_df):
    test_df.drop(single_val_features,axis =1,inplace=True)
    test_df.drop(['X4','ID'],axis=1,inplace= True)
    test_df.drop(single_val_feat_from_bin,axis =1,inplace= True)
    encode= joblib.load('cat_encode1.pkl')
    x_test_cat_encd = encode.transform(test_df[cat_list])
    
    x_test_cat_encd  =pd.DataFrame(x_test_cat_encd,columns=encode.get_feature_names()) 
    
    x_test_top_50_df  = x_test_cat_encd.iloc[:,kbest]
    test_df.drop(cat_list,axis=1,inplace = True)
    
    X_test_fin = pd.concat([test_df,x_test_top_50_df],axis=1)
    X_test_fin.head()
    
    return X_test_fin
 

#read test data  
test_data  = pd.read_csv(r"data\test.csv")
test_data.head()

X_test = preprocessing(test_data)


#predicting test data on model
best_regressor = joblib.load(r"model_deployment\xbg_best.pkl")
y_pred_test = best_regressor.predict(X_test.iloc[0:5,:])


print(y_pred_test)


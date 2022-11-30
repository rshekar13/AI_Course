from flask import Flask, jsonify, request
import pandas as pd
import pickle
import joblib
import sys
import io
import xgboost
import flask
app = Flask(__name__)

###########################################################
def preprocess():
    """ function to preprocess the given input to the data required for model"""
   
    req = request.form.to_dict()
    col = pd.read_csv(r'col_names.csv').columns
    drop_col = pd.read_csv(r'drop_col.csv').columns
    drop_col = [i.strip() for i in drop_col]
    cat_list = ['X0', 'X1', 'X2', 'X3', 'X5', 'X6', 'X8']
    query=  pd.read_csv(io.StringIO(req['review_text']), sep=",",names =col)
    query.drop(drop_col,axis=1,inplace=True)
    encode= joblib.load('cat_encode1.pkl')
    qvery_encd = encode.transform(query[cat_list])
    qvery_encd  =pd.DataFrame(qvery_encd,columns=encode.get_feature_names()) 
    kbest_indices= [  0,   1,   4,   7,   8,   9,  11,  14,  15,  17,  19,  22,  25,
        26,  30,  34,  35,  37,  38,  39,  42,  43,  44,  45,  52,  56,
        62,  71,  74,  75,  76,  80,  85,  87,  89,  99, 102, 103, 105,
        106, 110, 111, 112, 116, 118, 119, 121, 168, 169, 172]
    
    
    qvery_encd_50_df  = qvery_encd.iloc[:,kbest_indices]
    
    query.drop(cat_list,axis=1,inplace = True)
    
    query_fin = pd.concat([query,qvery_encd_50_df],axis=1)
    
    return query_fin



@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    final_df = preprocess()
    model = joblib.load("xbg_best.pkl")
    ypred = float(model.predict(final_df))
    ypred = round(ypred,2)
    print(ypred)
    #return jsonify({'prediction': ypred})
    return flask.render_template('predict.html',time=ypred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082,debug=True)

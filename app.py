from flask import Flask, render_template, request
from model import *
import pandas as pd 
app = Flask('__name__')


@app.route('/')
def view():
    return render_template('home.html')


@app.route('/recommend',methods=['POST'])
def recommend_top5():
    print(request.method)
    user_name = request.form['User Name']
    print('User name=',user_name)

    result = get_recommended_products(user_name)

    if isinstance(result, pd.DataFrame):
        return render_template('home.html',column_names=result.columns.values, row_data=list(result.values.tolist()), zip=zip, text='Recommended products')
    else:
        return render_template('home.html',text=result) 
    
    
if __name__ == '__main__':
    # app.debug=False
    app.run(debug=True)
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
    user_name = request.form['User']
    print('User name=',user_name)

    result = get_recommended_products(user_name)
    print(result)

    if isinstance(result, pd.DataFrame):
        # return render_template('home.html',column_names=result.columns.values, row_data=list(result.values.tolist()), zip=zip, text='Recommended products')
        return render_template('home.html', col_names=result.columns.values, 
                               products=result.name.to_list(),
                               brands=result.brand.to_list(),
                               manufacturers=result.manufacturer.to_list(),
                               text='Recommended Products') 
    else:
        return render_template('home.html', text=result) 
    
    
if __name__ == '__main__':
    # app.debug=False
    app.run(debug=True)
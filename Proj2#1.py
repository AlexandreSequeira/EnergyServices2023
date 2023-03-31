# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:47:33 2023

@author: Alexandre Sequeira
"""

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import plotly.graph_objects as go

#kBest
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)

#Ensemble methods
from sklearn.ensemble import RandomForestRegressor

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


#Load data
df = pd.read_csv('test_data_2019.csv')
dfx = pd.read_csv('Raw_data.csv')
df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type
dfx['Date'] = pd.to_datetime (dfx['Date'])

#Create missing variables
dfx['hour']= pd.Series(dfx['Date']).apply(lambda x: x.hour).values
dfx['Power-1']=dfx['Power_kW'].shift(1) # Previous hour consumption
df5x=dfx.dropna()

#Deleting outliers
df51x= df5x[df5x['Power_kW'] >df5x['Power_kW'].quantile(0.03)]
df52x=df51x[df51x['Power_kW'] <df5x['Power_kW'].quantile(0.99)]

#Define some future useful data
Z=df52x.values
Ypre=Z[:,9]
Y=Ypre.astype('float64')
Xpre=Z[:,[1,2,3,4,5,6,7,8,10,11]]
X=Xpre.astype('float64')

#Create KBest
features=SelectKBest(k=3,score_func=f_regression)
fit=features.fit(X,Y)
fig53=px.bar([i for i in range(len(fit.scores_))],fit.scores_,labels={'x': 'Value',
                                   'y':'Index'})

features=SelectKBest(k=2,score_func=f_regression)
fit=features.fit(X,Y)
fig52=px.bar([i for i in range(len(fit.scores_))],fit.scores_,labels={'x': 'Value',
                                   'y':'Index'})


#Create Random Forest
model = RandomForestRegressor()
model.fit(X, Y)
fig55=px.bar([i for i in range(len(model.feature_importances_))],
             model.feature_importances_,labels={'x': 'Value',
                                                'y':'Index'}) 


#Fig for previous data
df2x=dfx.iloc[:,2:10]
X2x=df2x.values
figx = px.line(dfx, x="Date", y=dfx.columns[1:12])
newnames = {'temp_C':'Temp[C]','HR':'HR[%]','windSpeed_m/s': 'WSpeed[m/s]',
            'windGust_m/s': 'WGust[m/s]','pres_mbar':'P[mbar]',
            'solarRad_W/m2': 'Radiation[W/m2]','rain_mm/h':'Rain[mm/h]',
            'rain_day': 'Rain day','Power_kW': 'Power[kW]','hour':'Hour',
            'Power-1': 'Power-1[kW]'}
figx.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )



#figxx= px.boxplot(x=dfx['Power_kW'])
#df_real=df.drop(['hour','Hollk','Power-1'],axis=1)
y2x=df2x['Power_kW'].values

df2=df.iloc[:,2:5]
X2=df2.values
fig = px.line(df, x="Date", y=df.columns[1:5])
df_real=df.drop(['hour','Hollk','Power-1'],axis=1)
y2=df_real['Power_kW'].values

#Load and run models


#Random Forest
with open('Models/RF_model.pkl','rb') as file:
    RF_model2=pickle.load(file)

y2_pred_RF = RF_model2.predict(X2)

###Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)

#Bootstrapping

with open('Models/BT_model.pkl','rb') as file:
    BT_model=pickle.load(file)

y2_pred_BT = BT_model.predict(X2)

###Evaluate errors
MAE_BT=metrics.mean_absolute_error(y2,y2_pred_BT) 
MSE_BT=metrics.mean_squared_error(y2,y2_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y2,y2_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y2)


#Gradient Boosting

with open('Models/GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)

y2_pred_GB = GB_model.predict(X2)

###Evaluate errors
MAE_GB=metrics.mean_absolute_error(y2,y2_pred_GB) 
MSE_GB=metrics.mean_squared_error(y2,y2_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y2)

#Linear Regression

with open('Models/LR_model.pkl','rb') as file:
    LR_model=pickle.load(file)

y2_pred_LR = LR_model.predict(X2)

###Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)


d = {'Methods': ['Linear Regression','Random Forest','Bootstrapping','Gradient Boosting'], 'MAE': [MAE_LR, MAE_RF,MAE_BT,MAE_GB], 'MSE': [MSE_LR, MSE_RF,MSE_BT,MSE_GB], 'RMSE': [RMSE_LR, RMSE_RF,RMSE_BT,RMSE_GB],'cvRMSE': [cvRMSE_LR, cvRMSE_RF,cvRMSE_BT,cvRMSE_GB]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df_real['Date'].values, 'Linear Regression': y2_pred_LR,'Random Forest': y2_pred_RF, 'Bootstrapping': y2_pred_BT,'Gradient Boosting': y2_pred_GB}
df_forecast=pd.DataFrame(data=d)
df_results=pd.merge(df_real,df_forecast, on='Date')



fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:8])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H2('IST Energy Forecast tool (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Model Errors', value='tab-3'),
        dcc.Tab(label='Previous Data', value='tab-4'),
        dcc.Tab(label='Feature Selection', value='tab-5')
        
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig,
            ),
            html.H6("""The meaning of the different values are:
                    Power_kW: average consumption of energy during that hour;
                    Power-1: average consumption in the previous hour;
                    Hollk:if August,Saturday,Sunday or holiday 1, else 0;
                    Hour: hour of the day;
                    """),
            
        ])
    if tab == 'tab-2':
        return html.Div([
            html.H3('IST Electricity Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
                ),
            
        ])
    if tab == 'tab-3':
        return html.Div([
            html.H3('Methods Errors'),
            generate_table(df_metrics)
        ])
    
    if tab == 'tab-4':
        return html.Div([
            html.H3('Explore previous data'),
                
            html.Label('Graph'),
            dcc.RadioItems(
                id='Radio41',
                options=[
                    {'label': 'Graphic display', 'value': 'Grp'},
                    {'label': 'Histogram', 'value': 'HIST'},  ],
                    inline= True,
                value='Grp',
            ),
            
            html.Label('Variable(only to histogram)'),
            dcc.RadioItems(
                id='Radio42',
                options=[
                    {'label': 'Temperature(temp_C)', 'value': 'temp'},
                    {'label': 'Relative humidity(HR)', 'value': 'HR'},
                    {'label': 'Wind Speed', 'value': 'winsp'},
                    {'label': 'Wind Gust', 'value': 'wingus'},
                    {'label': 'Pressure', 'value': 'pres'},
                    {'label': 'Solar Radiation', 'value': 'solrad'},
                    {'label': 'Rain Rate', 'value': 'rainrate'},
                    {'label': 'Rain Day', 'value': 'rainday'},
                    {'label': 'Power', 'value': 'P'},
                    {'label': 'Hour', 'value': 'hour'},
                    {'label': 'Power-1', 'value': 'P-1'},
                    ],
                    inline= True,
                value='P',
            ),
            
            dcc.Graph(
                id='graph4',
                figure=figx,
                ),
            ])
 
    elif tab == 'tab-5':
     return html.Div([
         html.H3('Feature Selection'),
         
         html.Label('Method'),
         dcc.RadioItems(
             id='Radio5',
             options=[
                 {'label': 'Kbest-3 variables', 'value': 'kb3'},
                 {'label': 'Random Forest', 'value': 'RF'},          ],
                 inline= True,
             value='kb3',
         ),
         
         dcc.Graph(
             id='graph5',
             figure=figx,
             ),
         html.H6("""The meaning of the different values are:
                 0:Temperature;
                 1:Relative humidity(HR);
                 2:Wind Speed;
                 3:Wind Gust;
                 4:Pressure;
                 5:Solar Radiation;
                 6:Rain Rate;
                 7:Rain Day;
                 8:Hour;
                 9:Power-1;
                 """),
     ])


@app.callback(
    dash.dependencies.Output('graph4', 'figure'),
     [dash.dependencies.Input('Radio41', 'value'),
      dash.dependencies.Input('Radio42','value')]
      
)
def update_graph(value1,value2):
    if value1 == 'HIST' :
        if value2 == 'P' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['Power_kW']))
            return figxx
        if value2 == 'HR' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['HR']))
            return figxx
        if value2 == 'winsp' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['windSpeed_m/s']))
            return figxx
        if value2 == 'wingus' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['windGust_m/s']))
            return figxx
        if value2 == 'pres' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['pres_mbar']))
            return figxx
        if value2 == 'solrad' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['solarRad_W/m2']))
            return figxx
        if value2 == 'rainrate' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['rain_mm/h']))
            return figxx
        if value2 == 'rainday' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['rain_day']))
            return figxx
        if value2 == 'temp' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['temp_C']))
            return figxx
        if value2 == 'hour' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['hour']))
            return figxx
        if value2 == 'P-1' :
            figxx= go.Figure()
            figxx.add_trace(go.Histogram(x=dfx['Power-1']))
            return figxx
        
    if value1 == 'Grp' :
        return figx
    
            
@app.callback(
    dash.dependencies.Output('graph5', 'figure'),
     dash.dependencies.Input('Radio5', 'value')
)

def update_graph2(value):
      if value == 'kb3' :
         return fig53
      if value == 'RF' :
         return fig55
    


if __name__ == '__main__':
    app.run_server()









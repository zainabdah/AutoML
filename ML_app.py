#importation des librairies
import base64
import datetime
import io
import base64
import datetime
import io
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import dash
from dash import no_update
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import dash_daq as daq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import cross_val_predict,cross_val_score, cross_validate,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from time import time
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import math
from sklearn.tree import DecisionTreeRegressor



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

########################################################################################################
###################################### Part 1: Layout #################################################

app.layout = html.Div([
    html.H3("Application d'apprentissage automatique",style={ 'textAlign': 'center','margin-top': '0px','font-weight': 'bold','font-size': '300%'}),
    html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Glisser-déposer ou ',
            html.A('Selectionner des Fichiers')
        ]),
        style={
            'width':'75%',
            'height': '30px',
            'lineHeight': '20px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'Left',
            'font-weight': 'bold',
            'font-size': 'large'
        },
         
        # téléchargement de plusieurs fichiers
        multiple=True
    ),
    html.P(id='flname',style={'font-weight': 'bold','font-size': 'large','color':'Green'}),
    html.Br(),
    html.P("Sélectionner la variable target",style={'font-weight': 'bold','font-size': 'large'}),
    html.Br(),
    dcc.Dropdown(id='target'),
    html.Hr(),
    html.P("Sélectionner les caractéristiques",style={'font-weight': 'bold','font-size': 'large'}),
    html.Br(),
    dcc.Dropdown(id='caracteristiques',multi=True),
    html.Br(),
    html.Button(id='bouton-soumission', children="Correlation Graph",style={'font-weight': 'bold','font-size': 'large'}),
    html.Br(),
    html.Div(id='Matrice-corr'),
    html.Br(),
    html.P("Choisir les algorithmes ML",style={'font-weight': 'bold','font-size': 'large'}),
    html.P(id='messageML',style={'font-weight': 'bold','color':'Blue'}),
    html.Br(),
    dcc.Dropdown(id='ML',multi=True),
    html.Hr(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Hr(),
],style={'width': "25%",'display': 'inline-block','background-color':'GhostWhite','margin-top':'0px','margin-bottom':'0px'}),
    
     html.Div([ 
    
     # ML Tables
     
     html.Div(id='div_onglets',children=[
     dcc.Tabs(id='tabs_onglets',style={'margin-top':'0%','padding': 20,'height':'50px'},children=[
     
     #1er ML algo table : regression logistique ou regression lineaire
     
     dcc.Tab(id='1er_ML',label='1er ML algo',style= {'background-color':'MediumSlateBlue','font-weight': 'bold','font-size': '110%'},
             children=[
                 
     #regression logistique : Liste des hyperparamètres manuels ou optimaux
     
     html.Div([
     html.Br(),
     html.Div(id='LGR_par_opt', children=
     [dcc.Dropdown(id='LGR_par_OM',placeholder="Réglage des paramètres",
     options=[
     {'label': y , 'value': y} for y in ['Parametres Manuels','Parametres optimaux']],
     style={'width':'90%'} 
     ),],style={'display': 'none'},className='six columns'),
     
     #Regression logistique : valeurs des hyperparametres 
     
         # Valeur de parametre C (parametres manuels)
         
     html.Div(id='LGR_C', children=[
     dcc.Dropdown(
     id='LGR_C_par',
     placeholder="Selectionner une valeur de C",
     options=[{'label':f'C={m}', 'value': m} for m in [0.01,0.1,1,10,100,500]],
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
     
         # Valeurs du Solver (parameteres manuels)
         
     html.Div(id='LGR_Solv', children=[
     dcc.Dropdown(
     id='LGR_Solv_par',
     placeholder="Select a Solver",
     options=[{'label':f'Solver={m}', 'value': m} for m in ['lbfgs','saga']],
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
     
         # Type de penalty (parameteres manuels)
         
     html.Div(id='LGR_pen', children=[
     dcc.Dropdown(id='LGR_pen_par',
     placeholder="Selectionner une penalty",
     options=[{'label':f'pelnaty={n}', 'value': n} for n in ['l1', 'l2']],
     multi=False,
     )], style= {'display': 'none'},className='six columns')
         
         
         ],className='row'),
     
     #Resultats et graphs de regression logistique
     html.Br(),
     html.Div(id='LGR_RES'),

    #La régression Ridge : hyperparameters manuels ou optimaux
    
    html.Div([
     html.Br(),
     html.Div(id='RR_par_opt', children=
     [dcc.Dropdown(id='RR_par_OM',placeholder="Réglage des paramètres",
     options=[
     {'label': y , 'value': y} for y in ['Parametres Manuels','Parametres Optimaux']],
     style={'width':'90%'} 
     ),],style={'display': 'none'},className='six columns'),
     
     #regression lineaire ridge: liste des valeurs d'hyper-parametres 
     
         # Valeurs du parametre Alpha (parametres manuels)
         
     html.Div(id='RR_alpha', children=[
     dcc.Dropdown(
     id='RR_alpha_par',
     placeholder="Selectionner la valeur alpha",
     options=[{'label':f'alpha={m}', 'value': m} for m in [0.01,0.1,1,10]],
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
     
         # Valeurs du Solver (parametres manuels)
         
     html.Div(id='RR_Solv', children=[
     dcc.Dropdown(
     id='RR_Solv_par',
     placeholder="Select a Solver",
     options=[{'label':f'Solver={m}', 'value': m} for m in ['auto','saga']],
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
     
         
         
         ],className='row'),
    
     #resultats et graphes de la regression lineaire ridge
     html.Br(),
     html.Div(id='RR_RES'),
     ]),
     
     #2eme ML algo table : Random Forest or KNN regressor 
     
     dcc.Tab(id='2eme_ML',label='2eme ML algo',style= {'background-color':'MediumSlateBlue','font-weight': 'bold','font-size': '110%'},
             children=[
     
     #Random Forest : Manual or optimal hyper-parameters
     
     html.Div([
     html.Br(),
     html.Div(id='RDF_par_opt', children=
     [dcc.Dropdown(id='RDF_par_OM',placeholder="Réglage des paramètres",
     options=[
     {'label': y , 'value': y} for y in ['Parametres Manuels','Parametres Optimaux']],
     style={'width':'90%'} 
     ),],style={'display': 'none'},className='five columns'),
     
     #Random Forest : hyper-parameters valeurs des  hyper-parameters
     
         # Nombre des estimateurs(parametres manuels)
         
     html.Div(id='RDF_est', children=[
     dcc.Dropdown(
     id='RDF_est_par',
     placeholder="Selectionner le nombre des arbres",
     options=[{'label':f'Trees number={m}', 'value': m} for m in [10,30,50,70,100,200,500]],
     multi=False,
     )],style= {'display': 'none'},className='five columns'),
     
         # Max depth (parametres manuels)
         
     html.Div(id='RDF_MXD', children=[
     dcc.Dropdown(
     id='RDF_MXD_par',
     placeholder="Selectionner max depth",
     options=[{'label':f'Max Depth={m}', 'value': m} for m in list(range(10,101,10))],
     multi=False,
     )],style= {'display': 'none'},className='five columns'),
     
         # Max features (parametres manuels)
     html.Div(id='RDF_MXF', children=[
     dcc.Dropdown(id='RDF_MXF_par',
     placeholder="Selectionner Max features",
     options=[{'label':f'Max feat={n}', 'value': n} for n in ['auto', 'sqrt']],
     multi=False,
     )], style= {'display': 'none'},className='five columns')
         
         
         ],className='row'), 
     
     #resultats de Random Forest 
     html.Br(),
     html.Div(id='RDF_RES'),

   #KNN:hyper-parametres Manuels ou optimals 
    
    html.Div([
     html.Br(),
     html.Div(id='KNN_par_opt', children=
     [dcc.Dropdown(id='KNN_par_OM',placeholder="Réglage de parametres",
     options=[
     {'label': y , 'value': y} for y in ['Manual Parameters','Optimal Parameters']],
     style={'width':'90%'} 
     ),],style={'display': 'none'},className='six columns'),
     
     #KNN : hyper-parameters values 
     
         # Number of neiighbors parameter (Manual parameters)
         
     html.Div(id='KNN_n', children=[
     dcc.Dropdown(
     id='KNN_n_par',
     placeholder="Select a Number of neighbors",
     options=[{'label':f'Number of neighbors={m}', 'value': m} for m in [1,3,5,7,9]],
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
     
         # Values of Solver (Manual parameters)
         
     html.Div(id='KNN_algo', children=[
     dcc.Dropdown(
     id='KNN_algo_par',
     placeholder="Select an algorithm",
     options=[{'label':f'algorithm={m}', 'value': m} for m in ['auto', 'ball_tree', 'kd_tree', 'brute']],
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
     
         
         
         ],className='row'), 
     #KNN graphs and results
     html.Br(),
     html.Div(id='KNN_RES'),
     ]),
     
     #3rd ML algo tab : SVM or Decision tree regressor
     
     dcc.Tab(id='3rd_ML',label='3rd ML algo',style= {'background-color':'MediumSlateBlue','font-weight': 'bold','font-size': '110%'},
             children=[
                 
    #SVM : Manual or optimal hyper-parameters
     
     html.Div([
     html.Br(),
     html.Div(id='SVM_par_opt', children=
     [dcc.Dropdown(id='SVM_par_OM',placeholder="Parameter Tuning",
     options=[
     {'label': y , 'value': y} for y in ['Manual Parameters','Optimal Parameters']],
     style={'width':'90%'} 
     ),],style={'display': 'none'},className='five columns'),
     
     #SVM : hyper-parameters values 
     
         # C values : penalty error (Manual parameters)
         
     html.Div(id='SVM_C', children=[
     dcc.Dropdown(
     id='SVM_C_par',
     placeholder="Select the penalty error",
     options=[{'label':f'C={m}', 'value': m} for m in [0.1,1,10,20]],
     multi=False,
     )],style= {'display': 'none'},className='five columns'),
     
         # Kernel options(Manual parameters)
         
     html.Div(id='SVM_KRN', children=[
     dcc.Dropdown(
     id='SVM_KRN_par',
     placeholder="Select a kernel",
     options=[{'label':f'Kernel={m}', 'value': m} for m in ["linear", "rbf", "poly"]],
     multi=False,
     )],style= {'display': 'none'},className='five columns'),
     
         # gammas values (Manual Parameters)
     html.Div(id='SVM_GAMMA', children=[
     dcc.Dropdown(id='SVM_GAMMA_par',
     placeholder="Select gammas values",
     options=[{'label':f'gamma={n}', 'value': n} for n in [0.1, 1, 10,20]],
     multi=False,
     )], style= {'display': 'none'},className='five columns')
         
         
         ],className='row'), 
     
     #SVM results
     html.Br(),
     html.Div(id='SVM_RES'),
     
     
     #Decision Tree Regressor : Manual or optimal hyper-parameters
     
     html.Div([
     html.Br(),
     html.Div(id='DTR_par_opt', children=
     [dcc.Dropdown(id='DTR_par_OM',placeholder="Parameter Tuning",
     options=[
     {'label': y , 'value': y} for y in ['Manual Parameters','Optimal Parameters']],
     style={'width':'90%'} 
     ),],style={'display': 'none'},className='five columns'),
     
     
     #DTR: hyper-parameters values 
     
         # criterion parameter (Manual parameters)

     html.Div(id='DTR_crit', children=[
     dcc.Dropdown(
     id='DTR_crit_par',
     placeholder="Select a criterion",
     options=[{'label':f'criterion={m}', 'value': m} for m in ["squared_error", "friedman_mse", "absolute_error", "poisson"]],
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
     
         # Max features(Manual parameters)
         
     html.Div(id='DTR_MXF', children=[
     dcc.Dropdown(
     id='DTR_MXF_par',
     placeholder="Select Max features",
     options=[{'label':f'Max_features={m}', 'value': m} for m in ['auto', 'sqrt']], # max number of features considered for splitting a node
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
   
     
         # Max depth (Manual parameters)
         
     html.Div(id='DTR_MXD', children=[
     dcc.Dropdown(
     id='DTR_MXD_par',
     placeholder="Select max depth",
     options=[{'label':f'Max Depth={m}', 'value': m} for m in list(range(10,101,10))],
     multi=False,
     )],style= {'display': 'none'},className='six columns'),
      ],className='row'),
     
     #DTR graphs and results
     html.Br(),
     html.Div(id='DTR_RES'),
     ]),
     ]),
     ]),
 ],style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top'})
],style={'background-color':'AliceBlue'})
     

############################################################################################
####################### Part 2: Upload button// Features and target ################################
#######################        ML algo dropdown/Correlation graph     ############################### 
############################################################################################

# Returns the dataframe that has been uploaded
def parse_contents(contents, filename):
    content_type, content_string = contents[0].split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename[0]:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename[0]:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

# Display file name and target variable
    
@app.callback(Output('flname', 'children'),
              Output('target', 'options'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def target(content,filename):
    df=parse_contents(content,filename)
    return filename,[{'label':x, 'value':x} for x in df.columns]

# Returns a list of ML algo according to the target variable type

def MList(df,value):
 out=[]
 if((is_numeric_dtype(df[value])) & (len(np.unique(df[str(value)]))>5) ):# si le nombre de modalités numériques est supérieur à 5 , on est pas sur un problème de classification
     out=['Linear Regression','KNN Regressor','Decision Tree Regressor']
 else:
     out=['Logistic Regression','Random Forest','SVM']
 return out

# Display ML algo according to the target variable type / Display features and type of target variable

@app.callback([Output('ML', 'options'),
              Output('features', 'options'),
              Output('messageML','children')],
              [Input('target','value'),
              Input('upload-data', 'contents')],
              State('upload-data', 'filename'))
def target_type(target,content,filename):
    df=parse_contents(content,filename)
    df1=df.drop(columns=[target])
    ML_list=MList(df,target)
    if((is_numeric_dtype(df[target])) & (len(np.unique(df[str(target)]))>5) ):# éviter qu'il considère des classes codées en 0 et 1 commme numériques
        return [{'label': x, 'value': x} for x in ML_list ], [{'label':x, 'value':x} for x in df1.columns],'Quantitative target: Regression problem'
    else:
        return [{'label': x, 'value': x} for x in ML_list ],[{'label':x, 'value':x} for x in df1.columns],'Qualitative target: Classification problem'


# Display the correlation graph 

@app.callback(Output('Corr-Matrix', 'children'),
              Input('submit-button','n_clicks'),
              Input('features','value'),
              Input('target','value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'))
def corelationMatrix(n,feat,target,content,filename):
    fig=html.Div()
    df=parse_contents(content,filename)
    feat.append(target)
    df_corr= df.loc[:,feat]
    corr_matrix=df_corr.corr()
    if n is None:
        return no_update
    else:
        fig = px.imshow(corr_matrix, title= "Corelation Matrix")
    return html.Div(children=[dcc.Graph(id='fig1', figure=fig)])

################################################################################################
################################### Part 3: TABS NAMES UPDATE  ##########################################
################################################################################################

# update the name of the 1st ML tab

@app.callback(Output('1st_ML', 'label'),
              [Input('target','value'),
              Input('upload-data', 'contents')],
              State('upload-data', 'filename'))
def update_1st_tab(target,content,filename):
    df=parse_contents(content,filename)
    ML_list=MList(df,target)
    if 'Linear Regression' in ML_list  :
        return 'Linear Regression'
    else :
            return 'Logistic Regression'

# update the name of the 2nd ML tab

@app.callback(Output('2nd_ML', 'label'),
              [Input('target','value'),
              Input('upload-data', 'contents')],
              State('upload-data', 'filename'))
def update_2nd_tab(target,content,filename):
    df=parse_contents(content,filename)
    ML_list=MList(df,target)
    if 'KNN Regressor' in ML_list  :
        return 'KNN Regressor'
    else :
        return 'Random Forest'

# update the name of the 3rd ML tab

@app.callback(Output('3rd_ML', 'label'),
              [Input('target','value'),
              Input('upload-data', 'contents')],
              State('upload-data', 'filename'))
def update_3rd_tab(target,content,filename):
    df=parse_contents(content,filename)
    ML_list=MList(df,target)
    if 'Decision Tree Regressor' in ML_list  :
        return 'Decision Tree Regressor'
    else :
        return 'SVM'



################################################################################################
#############################  Part 4: TABS Manual Hyper-parameters  ##########################################
################################################################################################

# 1st Tab: Logistic regression

    # Display a dropdown (optimal / Manual parameters) after selecting Logistic Regression in the ML dropDown

@app.callback(Output('LGR_par_opt', 'style'), 
 [Input('ML', 'value')]) 
 
def display_param(ML_selected): 
 
 style={'display': 'none'} 
 if 'Logistic Regression' in ML_selected: 
     style={'width': '20%','display': 'inline-block'} 
 
 return style 
 
    # Display a dropdown containing C values after selecting Manual parameters

@app.callback(Output('LGR_C', 'style'), 
 [Input('LGR_par_OM', 'value')], [Input('ML', 'value')]) 
 
def LGR_C_val(LGR_OM,ML_selected): 
 
 style={'display': 'none'} 
 if LGR_OM=='Manual Parameters': 
     if 'Logistic Regression' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 
 return style 

    # Display a dropdown containing Solver options after selecting Manual parameters

@app.callback(Output('LGR_Solv', 'style'), 
 [Input('LGR_par_OM', 'value')], [Input('ML', 'value')]) 
 
def LGR_Solv_opt(LGR_OM,ML_selected): 
 
 style={'display': 'none'} 
 if LGR_OM=='Manual Parameters': 
     if 'Logistic Regression' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style 

    # Display a dropdown containing penalty options after selecting Manual parameters

@app.callback(Output('LGR_pen', 'style'), 
 [Input('LGR_par_OM', 'value')], [Input('ML', 'value')]) 
 
def LGR_pen_opt(LGR_OM,ML_selected): 
 
 style={'display': 'none'} 
 if LGR_OM=='Manual Parameters': 
     if 'Logistic Regression' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style 

@app.callback(Output('ML', 'value'), [Input('ML', 'options')])
def reinitialiser_1(value):
 return ""
@app.callback(Output('LGR_results', 'value'), [Input('LGR_results', 'children')])
def callback16(value):
 return ""



# transform classification report to dataframe

def transf_df(report):
 report = [x.split(' ') for x in report.split('\n')]
 header = ['Class']+['Precision']+['Recall']+['F score']+['Fréquence']
 values = []
 for row in report[1:-5]:
     row = [value for value in row if value!='']
 if row!=[]:
     values.append(row)
     df = pd.DataFrame(data = values, columns = header)
 return df


# 1st tab : Ridge linear regression

 # Display a dropdown (optimal / Manual parameters) after selecting Linear Ridge Regression in the ML dropDown

@app.callback(Output('RR_par_opt', 'style'), 
 [Input('ML', 'value')]) 
 
def display_param(ML_selected): 
 
 style={'display': 'none'} 
 if 'Linear Regression' in ML_selected: 
     style={'width': '20%','display': 'inline-block'} 
 
 return style 

 # Display a dropdown containing alpha values after selecting Manual parameters

@app.callback(Output('RR_alpha', 'style'), 
 [Input('RR_par_OM', 'value')], [Input('ML', 'value')]) 
 
def LGR_C_val(RR_OM,ML_selected): 
 
 style={'display': 'none'} 
 if RR_OM=='Manual Parameters': 
     if 'Linear Regression' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 
 return style 

    # Display a dropdown containing Solver options after selecting Manual parameters

@app.callback(Output('RR_Solv', 'style'), 
 [Input('RR_par_OM', 'value')], [Input('ML', 'value')]) 
 
def LGR_Solv_opt(RR_OM,ML_selected): 
 
 style={'display': 'none'} 
 if RR_OM=='Manual Parameters': 
     if 'Linear Regression' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style 



# 2nd Tab: Random Forest

    # Display a dropdown (optimal / Manual parameters) after selecting Random Forest in the ML dropDown

@app.callback(Output('RDF_par_opt', 'style'), 
 [Input('ML', 'value')]) 
 
def display_param(ML_selected): 
 style={'display': 'none'} 
 if 'Random Forest' in ML_selected: 
     style={'width': '20%','display': 'inline-block'} 
 
 return style 
 
    # Display a dropdown containing Number of trees values after selecting Manual parameters

@app.callback(Output('RDF_est', 'style'), 
 [Input('RDF_par_OM', 'value')], [Input('ML', 'value')]) 
 
def RDF_est_val(RDF_OM,ML_selected): 
 
 style={'display': 'none'} 
 if RDF_OM=='Manual Parameters': 
     if 'Random Forest' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 
 return style 

    # Display a dropdown containing Max Depth options after selecting Manual parameters

@app.callback(Output('RDF_MXD', 'style'), 
 [Input('RDF_par_OM', 'value')], [Input('ML', 'value')]) 
 
def RDF_MXD_opt(RDF_OM,ML_selected): 
 
 style={'display': 'none'} 
 if RDF_OM=='Manual Parameters': 
     if 'Random Forest' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style 

    # Display a dropdown containing Max features options after selecting Manual parameters

@app.callback(Output('RDF_MXF', 'style'), 
 [Input('RDF_par_OM', 'value')], [Input('ML', 'value')]) 
 
def RDF_MXF_opt(RDF_OM,ML_selected): 
 
 style={'display': 'none'} 
 if RDF_OM=='Manual Parameters': 
     if 'Random Forest' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style 

# 2nd tab : KNN

 # Display a dropdown (optimal / Manual parameters) after selecting KNN in the ML dropDown

@app.callback(Output('KNN_par_opt', 'style'), 
 [Input('ML', 'value')]) 
 
def display_param(ML_selected): 
 
 style={'display': 'none'} 
 if 'KNN Regressor' in ML_selected: 
     style={'width': '20%','display': 'inline-block'} 
 
 return style 

 # Display a dropdown containing number of neighbors after selecting Manual parameters

@app.callback(Output('KNN_n', 'style'), 
 [Input('KNN_par_OM', 'value')], [Input('ML', 'value')]) 
 
def KNN_n_val(KNN_OM,ML_selected): 
 
 style={'display': 'none'} 
 if KNN_OM=='Manual Parameters': 
     if 'KNN Regressor'in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 
 return style 

    # Display a dropdown containing algorithm options after selecting Manual parameters

@app.callback(Output('KNN_algo', 'style'), 
 [Input('KNN_par_OM', 'value')], [Input('ML', 'value')]) 
 
def KNN_algo_opt(KNN_OM,ML_selected): 
 
 style={'display': 'none'} 
 if KNN_OM=='Manual Parameters': 
     if 'KNN Regressor' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style



# 3rd Tab: SVM

    # Display a dropdown (optimal / Manual parameters) after selecting SVM in the ML dropDown

@app.callback(Output('SVM_par_opt', 'style'), 
 [Input('ML', 'value')]) 
 
def display_param(ML_selected): 
 style={'display': 'none'} 
 if 'SVM' in ML_selected: 
     style={'width': '20%','display': 'inline-block'} 
 
 return style 
 
    # Display a dropdown containing C values after selecting Manual parameters

@app.callback(Output('SVM_C', 'style'), 
 [Input('SVM_par_OM', 'value')], [Input('ML', 'value')]) 
 
def SVM_C_val(SVM_OM,ML_selected): 
 
 style={'display': 'none'} 
 if SVM_OM=='Manual Parameters': 
     if 'SVM' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 
 return style 

    # Display a dropdown containing kernel options after selecting Manual parameters

@app.callback(Output('SVM_KRN', 'style'), 
 [Input('SVM_par_OM', 'value')], [Input('ML', 'value')]) 
 
def SVM_kernel_opt(SVM_OM,ML_selected): 
 
 style={'display': 'none'} 
 if SVM_OM=='Manual Parameters': 
     if 'SVM' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style 

    # Display a dropdown containing gammas values after selecting Manual parameters

@app.callback(Output('SVM_GAMMA', 'style'), 
 [Input('SVM_par_OM', 'value')], [Input('ML', 'value')]) 
 
def SVM_gamma_opt(SVM_OM,ML_selected): 
 
 style={'display': 'none'} 
 if SVM_OM=='Manual Parameters': 
     if 'SVM' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style 



# 3rd tab : DTR

 # Display a dropdown (optimal / Manual parameters) after selecting DTR in the ML dropDown

@app.callback(Output('DTR_par_opt', 'style'), 
 [Input('ML', 'value')]) 
 
def display_param(ML_selected): 
 
 style={'display': 'none'} 
 if 'Decision Tree Regressor' in ML_selected: 
     style={'width': '20%','display': 'inline-block'} 
 
 return style 

 # Display a dropdown containing criterion after selecting Manual parameters

@app.callback(Output('DTR_crit', 'style'), 
 [Input('DTR_par_OM', 'value')], [Input('ML', 'value')]) 
 
def DTR_crit_val(DTR_OM,ML_selected): 
 
 style={'display': 'none'} 
 if DTR_OM=='Manual Parameters': 
     if 'Decision Tree Regressor'in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 
 return style 

    # Display a dropdown containing Max features options after selecting Manual parameters

@app.callback(Output('DTR_MXF', 'style'), 
 [Input('DTR_par_OM', 'value')], [Input('ML', 'value')]) 
 
def DTR_MXFR_opt(DTR_OM,ML_selected): 
 
 style={'display': 'none'} 
 if DTR_OM=='Manual Parameters': 
     if 'Decision Tree Regressor' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style


    # Display a dropdown containing Max depth values after selecting Manual parameters

@app.callback(Output('DTR_MXD', 'style'), 
 [Input('DTR_par_OM', 'value')], [Input('ML', 'value')]) 
 
def DTR_MXDR_opt(DTR_OM,ML_selected): 
 
 style={'display': 'none'} 
 if DTR_OM=='Manual Parameters': 
     if 'Decision Tree Regressor' in ML_selected: 
         style={'width': '20%','display': 'inline-block'} 
 return style


#############################################################################################
############################################################################################
####################### Part 5 : Models Results   ################################
############################################################################################


#########################################################################################
############################ Regression logistics results ################################
#########################################################################################


@app.callback(Output('LGR_RES', 'children'),
              Input('target', 'value'),
              Input('features','value'),
              Input('LGR_par_OM', 'value'),
              Input('LGR_C_par','value'),
              Input('LGR_Solv_par','value'),
              Input('LGR_pen_par','value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))

def update_LGR(target,features,LGR_OM,LGR_C,LGR_Solv,LGR_pen,content,filename):
     if content and features and target and LGR_OM :
         df=parse_contents(content,filename)
         start=time() 
         X=df.loc[:,features]
         X=pd.get_dummies(X)
         sc=StandardScaler() 
         X=pd.DataFrame(sc.fit_transform(X))
         y=df[str(target)]
         
         if LGR_OM=='Manual Parameters':
             if LGR_C:
                 if LGR_Solv:
                     if LGR_pen:
                         coef=LGR_C
                         solv=LGR_Solv
                         pen=LGR_pen
         if LGR_OM=='Optimal Parameters':
             par=[{'C':[0.01,0.1,1,10,100,500],'solver' : ['lbfgs','saga'],'penalty' : ['l1', 'l2']}]
             LGR = LogisticRegression()
             clf = GridSearchCV(LGR, param_grid = par, cv = 10, verbose=True, n_jobs=-1, scoring='accuracy')
             clf.fit(X,y)
             best_par=list(clf.best_params_.values())
             coef=best_par[0]
             solv=best_par[2]
             pen=best_par[1]
        
      
         ModelLGR = LogisticRegression(C=coef,solver=solv,penalty=pen,max_iter=100000)
         predicted = cross_val_predict(ModelLGR, X, y, cv=10)
         predicted_proba= cross_val_predict(ModelLGR, X, y, cv=10,method='predict_proba')             
         
                      
         acc=accuracy_score(y,predicted)
         acc=round(acc,3)
         
         cnf_matrix = metrics.confusion_matrix(y, predicted)
         confusion_matrix = pd.crosstab(y, predicted, rownames=['Actual'], colnames=['Predicted'])
         
         clreport=metrics.classification_report(y,predicted)
         clreport=transf_df(clreport)
         
         fig_matcon=px.imshow(cnf_matrix,labels=dict(x="Predicted", y="Actual", color="Individuals"),x=np.unique(predicted),y=np.unique(y),color_continuous_scale='Inferno',title="Confusion_Matrix")
         
         fig_ROC = go.Figure()
         fig_ROC.add_shape(
                             type='line', line=dict(dash='dot'),fillcolor='black',
                             x0=0, x1=1, y0=0, y1=1
                             ) 
    
         
         
         if predicted_proba.shape[1]==2:
             lb = LabelEncoder() 
             y=lb.fit_transform(y)
             fpr, tpr, threshold = roc_curve(y,predicted_proba[:,1],pos_label=1)
             auc = metrics.roc_auc_score(y,predicted_proba[:,1])
             auc=round(auc,3)
             roc_name = f"AUC is {auc}"
            
             fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr,name=roc_name,mode='lines',line_color='red'))
         else:
             y_onehot = pd.get_dummies(y)
             for i in range(predicted_proba.shape[1]):
                 y_true = y_onehot.iloc[:, i]
                 
                 y_score = predicted_proba[:, i]
                 
                 fpr, tpr, _ = roc_curve(y_true, y_score)
                 auc_score = roc_auc_score(y_true, y_score)
                 name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                 fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        
         fig_ROC.update_layout(
             xaxis_title='False positive rate',
             yaxis_title='True positive rate',
             yaxis=dict(scaleanchor="x", scaleratio=1),
             xaxis=dict(constrain='domain'),
             width=700, height=500,
             title='ROC Curve',
             showlegend=True
             )
         fig_thresh=go.Figure()
         
         end=time()
         dur=(end-start)
         dur=round(dur,3)
      

         return html.Div([html.H2(children=f"Computation time = {dur}",
                                  style={'textAlign': 'center'
                                            }),
                          html.Div(children=f"Accuracy = {acc}", style={
                                            'textAlign': 'center'
                                            }),
                          html.Div([dash_table.DataTable(id='data-table',
                                            columns=[{"name": i, "id": i} for i in clreport.columns],
                                            data=clreport.to_dict('rows'),
                                            editable=True,
                                            style_header={
                                                'backgroundColor': 'black',
                                                'color': 'white',
                                                'textAlign':'center',
                                                'font-weight': 'bold','font-size': '200%'
                                                },
                                           style_data={
                                           'backgroundColor': 'black',
                                           'color': 'white','textAlign':'center',
                                           'font-weight': 'bold','font-size': '110%'
                                               },
                                            )]),
                          html.Div([dcc.Graph(id='MaCo', figure=fig_matcon)]),
                          html.Div([dcc.Graph(id='ROC', figure=fig_ROC)
                          ])
                          ])
         

#########################################################################################
############################ Random Forest results ################################
#########################################################################################

                                

@app.callback(Output('RDF_RES', 'children'),
              Input('target', 'value'),
              Input('features','value'),
              Input('RDF_par_OM', 'value'),
              Input('RDF_est_par','value'),
              Input('RDF_MXD_par','value'),
              Input('RDF_MXF_par','value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))

def update_RDF(target,features,RDF_OM,RDF_est,RDF_MXD,RDF_MXF,content,filename):
     if content and features and target and RDF_OM :
         df=parse_contents(content,filename)
         start=time() 
         X=df.loc[:,features]
         X=pd.get_dummies(X)
         sc=StandardScaler() 
         X=pd.DataFrame(sc.fit_transform(X))
         y=df[str(target)]
         if RDF_OM=='Manual Parameters':
             if RDF_est:
                 if RDF_MXD:
                     if RDF_MXF:
                         n_est=RDF_est
                         MXD=RDF_MXD
                         MXF=RDF_MXF
                         MSP=5
                         MSL=2
                         bst=True
         if RDF_OM=='Optimal Parameters':
             param_dist = {"n_estimators":[10,30,50,70,100,200,500], # Number of trees in random forest
                           "max_depth": list(range(10,101,10)), # max number of levels in each decision tree
                           "max_features": ['auto', 'sqrt'], # max number of features considered for splitting a node
                           "min_samples_split": [2,5,10], # min number of data points placed in a node before the node is split
                           "min_samples_leaf": [1, 2, 4],# min number of data points allowed in a leaf node
                           "bootstrap": [True, False], # method for sampling data points (with or without replacement)
                  
                  }
             forest_reg = RandomForestClassifier()
             rand_search = RandomizedSearchCV(forest_reg, param_dist, cv=10, scoring='accuracy', verbose=True, n_jobs=-1)
             rand_search.fit(X,y)
             best_par=list(rand_search.best_params_.values())
             n_est=best_par[4]
             MXD=best_par[0]
             MXF=best_par[3]
             MSP=best_par[1]
             MSL=best_par[2]
             bst=best_par[5]
        
         ModelRDF = RandomForestClassifier(n_estimators=n_est,max_depth=MXD,max_features=MXF,min_samples_split=MSP,min_samples_leaf=MSL,bootstrap=bst)
         predicted = cross_val_predict(ModelRDF, X, y, cv=10)
         predicted_proba= cross_val_predict(ModelRDF, X, y, cv=10,method='predict_proba')             
       
                      
         acc=accuracy_score(y,predicted)
         acc=round(acc,3)
         
         cnf_matrix = metrics.confusion_matrix(y, predicted)
         confusion_matrix = pd.crosstab(y, predicted, rownames=['Actual'], colnames=['Predicted'])
       
         clreport=metrics.classification_report(y,predicted)
         clreport=transf_df(clreport)
        
         fig_matcon=px.imshow(cnf_matrix,labels=dict(x="Predicted", y="Actual", color="Individuals"),x=np.unique(predicted),y=np.unique(y),color_continuous_scale='Inferno',title="Confusion_Matrix")
         
         fig_ROC = go.Figure()
         fig_ROC.add_shape(
                             type='line', line=dict(dash='dot'),fillcolor='black',
                             x0=0, x1=1, y0=0, y1=1
                             ) 
    
         
         
         if predicted_proba.shape[1]==2:
             lb = LabelEncoder() 
             y=lb.fit_transform(y)
             fpr, tpr, threshold = roc_curve(y,predicted_proba[:,1],pos_label=1)
             auc = metrics.roc_auc_score(y,predicted_proba[:,1])
             auc=round(auc,3)
             
             roc_name = f"AUC is {auc}"
             
             fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr,name=roc_name,mode='lines',line_color='red'))
         else:
             y_onehot = pd.get_dummies(y)
             for i in range(predicted_proba.shape[1]):
                 y_true = y_onehot.iloc[:, i]
                
                 y_score = predicted_proba[:, i]
                 
                 fpr, tpr, _ = roc_curve(y_true, y_score)
                 auc_score = roc_auc_score(y_true, y_score)
                 name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                 fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
       
         fig_ROC.update_layout(
             xaxis_title='False positive rate',
             yaxis_title='True positive rate',
             yaxis=dict(scaleanchor="x", scaleratio=1),
             xaxis=dict(constrain='domain'),
             width=700, height=500,
             title='ROC Curve',
             showlegend=True
             )
         fig_thresh=go.Figure()
         
         end=time()
         dur=(end-start)
         dur=round(dur,3)
         

         return html.Div([html.H2(children=f"Computation time = {dur}",
                                  style={'textAlign': 'center'
                                            }),
                          html.Div(children=f"Accuracy = {acc}", style={
                                            'textAlign': 'center'
                                            }),
                          html.Div([dash_table.DataTable(id='data-table_1',
                                            columns=[{"name": i, "id": i} for i in clreport.columns],
                                            data=clreport.to_dict('rows'),
                                            editable=True,
                                            style_header={
                                                'backgroundColor': 'black',
                                                'color': 'white',
                                                'textAlign':'center',
                                                'font-weight': 'bold','font-size': '200%'
                                                },
                                           style_data={
                                           'backgroundColor': 'black',
                                           'color': 'white','textAlign':'center',
                                           'font-weight': 'bold','font-size': '110%'
                                               },
                                            )]),
                          html.Div([dcc.Graph(id='MaCo', figure=fig_matcon)]),
                          html.Div([dcc.Graph(id='ROC', figure=fig_ROC)
                          ])
                          ])
         




#########################################################################################
############################ SVM results ################################
#########################################################################################


                                   # Results details

@app.callback(Output('SVM_RES', 'children'),
              Input('target', 'value'),
              Input('features','value'),
              Input('SVM_par_OM', 'value'),
              Input('SVM_C_par','value'),
              Input('SVM_KRN_par','value'),
              Input('SVM_GAMMA_par','value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))

def update_SVM(target,features,SVM_OM,SVM_C,SVM_KRN,SVM_GAMMA,content,filename):
    if content and features and target and SVM_OM :
         df=parse_contents(content,filename)
         start=time() 
         X=df.loc[:,features]
         X=pd.get_dummies(X)
         sc=StandardScaler() 
         X=pd.DataFrame(sc.fit_transform(X))
         y=df[str(target)]
         if SVM_OM=='Manual Parameters':
             if SVM_C:
                 if SVM_KRN:
                     if SVM_GAMMA:
                         C_par=SVM_C
                         KRN_par=SVM_KRN
                         Gamma_par=SVM_GAMMA
         if SVM_OM=='Optimal Parameters':
             param_dist = {"C": [0.1,1,10], # C controls the trade off between smooth decision boundary and classifying the training points correctly.
                           "kernel": ["linear", "rbf","poly"], # type of hyperplane used to separate the data ( rbf and poly are non linear hyperplanes)
                           
                  }
             svc_reg = SVC()
             svc_search = GridSearchCV(svc_reg, param_dist, cv=10, scoring='accuracy', verbose=True, n_jobs=-1)
             svc_search.fit(X,y)
             best_par=list(svc_search.best_params_.values())
           
             C_par=best_par[0]
             KRN_par=best_par[1]
             
        
         ModelSVM = SVC(C=C_par,kernel=KRN_par,probability=True)
         predicted = cross_val_predict(ModelSVM, X, y, cv=10)
         
         predicted_proba= cross_val_predict(ModelSVM, X, y, cv=10,method='predict_proba')
         acc=accuracy_score(y,predicted)
         acc=round(acc,3)
         
         cnf_matrix = metrics.confusion_matrix(y, predicted)
         confusion_matrix = pd.crosstab(y, predicted, rownames=['Actual'], colnames=['Predicted'])
         
         clreport=metrics.classification_report(y,predicted)
         clreport=transf_df(clreport)
         
         fig_matcon=px.imshow(cnf_matrix,labels=dict(x="Predicted", y="Actual", color="Individuals"),x=np.unique(predicted),y=np.unique(y),color_continuous_scale='Inferno',title="Confusion_Matrix")
         
         fig_ROC = go.Figure()
         fig_ROC.add_shape(
                             type='line', line=dict(dash='dot'),fillcolor='black',
                             x0=0, x1=1, y0=0, y1=1
                             ) 
   
         
         if predicted_proba.shape[1]==2:
             lb = LabelEncoder() 
             y=lb.fit_transform(y)
             fpr, tpr, threshold = roc_curve(y,predicted_proba[:,1],pos_label=1)
             auc = metrics.roc_auc_score(y,predicted_proba[:,1])
             auc=round(auc,3)
          
             roc_name = f"AUC is {auc}"
          
             fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr,name=roc_name,mode='lines',line_color='red'))
         else:
             y_onehot = pd.get_dummies(y)
             for i in range(predicted_proba.shape[1]):
                 y_true = y_onehot.iloc[:, i]
                
                 y_score = predicted_proba[:, i]
                
                 fpr, tpr, _ = roc_curve(y_true, y_score)
                 auc_score = roc_auc_score(y_true, y_score)
                 name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                 fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
       
         fig_ROC.update_layout(
             xaxis_title='False positive rate',
             yaxis_title='True positive rate',
             yaxis=dict(scaleanchor="x", scaleratio=1),
             xaxis=dict(constrain='domain'),
             width=700, height=500,
             title='ROC Curve',
             showlegend=True
             )
         fig_thresh=go.Figure()
         
         end=time()
         dur=(end-start)
         dur=round(dur,3)
         

         return html.Div([html.H2(children=f"Computation time = {dur}",
                                  style={'textAlign': 'center'
                                            }),
                          html.Div(children=f"Accuracy = {acc}", style={
                                            'textAlign': 'center'
                                            }),
                          html.Div([dash_table.DataTable(id='data-table_1',
                                            columns=[{"name": i, "id": i} for i in clreport.columns],
                                            data=clreport.to_dict('rows'),
                                            editable=True,
                                            style_header={
                                                'backgroundColor': 'black',
                                                'color': 'white',
                                                'textAlign':'center',
                                                'font-weight': 'bold','font-size': '200%'
                                                },
                                           style_data={
                                           'backgroundColor': 'black',
                                           'color': 'white','textAlign':'center',
                                           'font-weight': 'bold','font-size': '110%'
                                               },
                                            )]),
                          html.Div([dcc.Graph(id='MaCo', figure=fig_matcon)]),
                          html.Div([dcc.Graph(id='ROC', figure=fig_ROC)
                          ])
                          ])
         




#########################################################################################
############################ Linear Ridge Regression results ################################
#########################################################################################

                                # Results details

@app.callback(Output('RR_RES', 'children'),
              Input('target', 'value'),
              Input('features','value'),
              Input('RR_par_OM', 'value'),
              Input('RR_alpha_par','value'),
              Input('RR_Solv_par','value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))

def update_RR(target,features,RR_OM,RR_alpha,RR_Solv,content,filename):
     if content and features and target and RR_OM :
         df=parse_contents(content,filename)
         start=time() 
         X=df.loc[:,features]
         X=pd.get_dummies(X)
         sc=StandardScaler() 
         X=pd.DataFrame(sc.fit_transform(X))
         y=df[str(target)]
         
         if RR_OM=='Manual Parameters':
             if RR_alpha:
                 if RR_Solv:
                         regul=RR_alpha
                         solv=RR_Solv
         if RR_OM=='Optimal Parameters':
             par=[{'alpha':[0.01,0.1,1,10],'solver' : ['auto','saga']}]
             RR = Ridge()
             clf = GridSearchCV(RR, param_grid = par, cv = 10, verbose=True, n_jobs=-1, scoring='neg_mean_squared_error')
             clf.fit(X,y)
             best_par=list(clf.best_params_.values())
             regul=best_par[0]
             solv=best_par[1]
        
         ModelRR = Ridge(alpha=regul,solver=solv)
         predicted = cross_val_predict(ModelRR, X, y, cv=10)
         mse=mean_squared_error(y,predicted)
         rmse=math.sqrt(mse)
         rmse=round(rmse,2)
         r2=r2_score(y,predicted)
         r2=round(r2,2)
         end=time()
         dur=(end-start)
         dur=round(dur,3)
    
         fig = go.Figure()
         fig.add_trace(go.Scatter(x=y, y=predicted,name='Regression Fit',mode='markers'))
         
         fig.update_layout(
             xaxis_title='Measured',
             yaxis_title='Predicted',
             width=700, height=500
             )
         fig.add_shape(
                             type='line', line=dict(dash='dot'),fillcolor='red',
                             x0=y.min(), x1=y.max(), y0=y.min(), y1=y.max()
                             ) 
         return html.Div([html.H4(children=f"Computation time = {dur}",
                                  style={'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Black',
                                            }),
                          html.Br(),
                          html.H4(children=f"RMSE = {rmse}", style={
                                            'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Green',
                                            }),
                          html.Br(),
                          html.H4(children=f"R squared= {r2}", style={
                                            'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Orange',
                                            }),
                          html.Div([dcc.Graph(id='ROC', figure=fig)]),
                     
                          ])
#########################################################################################
############################ KNN results ################################
#########################################################################################

                                # Results details

@app.callback(Output('KNN_RES', 'children'),
              Input('target', 'value'),
              Input('features','value'),
              Input('KNN_par_OM', 'value'),
              Input('KNN_n_par','value'),
              Input('KNN_algo_par','value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))

def update_KNN(target,features,KNN_OM,KNN_n,KNN_algo,content,filename):
     if content and features and target and KNN_OM :
         df=parse_contents(content,filename)
         start=time() 
         X=df.loc[:,features]
         X=pd.get_dummies(X)
         sc=StandardScaler() 
         X=pd.DataFrame(sc.fit_transform(X))
         y=df[str(target)]
         
         if KNN_OM=='Manual Parameters':
             if KNN_n:
                 if KNN_algo:
                         neighbors=KNN_n
                         solv=KNN_algo
         if KNN_OM=='Optimal Parameters':
             par=[{'n_neighbors':[1,3,5,7,9],'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}]
             knn = KNeighborsRegressor()
             clf = GridSearchCV(knn, param_grid = par, cv = 10, verbose=True, n_jobs=-1, scoring='neg_mean_squared_error')
             clf.fit(X,y)
             best_par=list(clf.best_params_.values())
             neighbors=best_par[1]
             solv=best_par[0]
        
         Modelknn = KNeighborsRegressor(n_neighbors=neighbors,algorithm=solv)
         predicted = cross_val_predict(Modelknn, X, y, cv=10)
         mse=mean_squared_error(y,predicted)
         rmse=math.sqrt(mse)
         rmse=round(rmse,2)
         r2=r2_score(y,predicted)
         r2=round(r2,2)
         end=time()
         dur=(end-start)
         dur=round(dur,3)
         fig = go.Figure()
         
         fig.add_trace(go.Scatter(x=y, y=predicted,name='predicted vs measured',mode='markers'))
         
         fig.update_layout(
             xaxis_title='Measured',
             yaxis_title='Predicted',
             width=700, height=500
             )
         fig.add_shape(
                             type='line', line=dict(dash='dot',color='DarkSlateGrey'),
                             x0=y.min(), x1=y.max(), y0=predicted.min(), y1=predicted.max()
                             ) 
       
         return html.Div([html.H4(children=f"Computation time = {dur}",
                                  style={'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Black',
                                            }),
                          html.Br(),
                          html.H4(children=f"RMSE = {rmse}", style={
                                            'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Green',
                                            }),
                          html.Br(),
                          html.H4(children=f"R squared= {r2}", style={
                                            'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Orange',
                                            }),
                          html.Div([dcc.Graph(id='ROC', figure=fig)]),
                     
                          ])


#########################################################################################
############################ Decision Tree results ################################
#########################################################################################

                                # Results details

@app.callback(Output('DTR_RES', 'children'),
              Input('target', 'value'),
              Input('features','value'),
              Input('DTR_par_OM', 'value'),
              Input('DTR_crit_par','value'),
              Input('DTR_MXF_par','value'),
              Input('DTR_MXD_par','value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))

def update_DTR(target,features,DTR_OM,DTR_crit,DTR_MXF,DTR_MXD,content,filename):
     if content and features and target and DTR_OM :
         df=parse_contents(content,filename)
         start=time() 
         X=df.loc[:,features]
         X=pd.get_dummies(X)
         sc=StandardScaler() 
         X=pd.DataFrame(sc.fit_transform(X))
         y=df[str(target)]
         
         if DTR_OM=='Manual Parameters':
             if DTR_crit:
                 if DTR_MXF:
                     if DTR_MXD:
                         criterion=DTR_crit
                         max_features=DTR_MXF
                         max_depth=DTR_MXD
         if DTR_OM=='Optimal Parameters':
             par=[{'criterion':["squared_error", "friedman_mse", "absolute_error"],"max_depth": list(range(10,101,10)), # max number of levels in the decision tree
                               "max_features": ['auto', 'sqrt'], # max number of features considered for splitting a node
                               }]
             DTR = DecisionTreeRegressor()
             clf = GridSearchCV(DTR, param_grid = par, cv = 10, verbose=True, n_jobs=-1, scoring='neg_mean_squared_error')
             clf.fit(X,y)
             best_par=list(clf.best_params_.values())
             criterion=best_par[0]
             
             max_depth=best_par[1]
             
             max_features=best_par[2]
           
             
        
         ModelDTR = DecisionTreeRegressor(criterion=criterion,max_depth=max_depth,max_features=max_features)
         predicted = cross_val_predict(ModelDTR, X, y, cv=10)
         mse=mean_squared_error(y,predicted)
         rmse=math.sqrt(mse)
         rmse=round(rmse,2)
         r2=r2_score(y,predicted)
         r2=round(r2,2)
         end=time()
         dur=(end-start)
         dur=round(dur,3)
       
         fig = go.Figure()
         fig.add_trace(go.Scatter(x=y, y=predicted,name='predicted vs measured',mode='markers'))
         
         fig.update_layout(
             xaxis_title='Measured',
             yaxis_title='Predicted',
             width=700, height=500
             )
         fig.add_shape(
                             type='line', line=dict(dash='dot'),fillcolor='red',
                             x0=y.min(), x1=y.max(), y0=predicted.min(), y1=predicted.max()
                             ) 
         
         return html.Div([html.H4(children=f"Computation time = {dur}",
                                  style={'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Black',
                                            }),
                          html.Br(),
                          html.H4(children=f"RMSE = {rmse}", style={
                                            'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Green',
                                            }),
                          html.Br(),
                          html.H4(children=f"R squared= {r2}", style={
                                            'textAlign': 'center','font-weight': 'bold','font-size': 'large','color':'Orange',
                                            }),
                          html.Div([dcc.Graph(id='ROC', figure=fig)]),
                     
                          ])

app.run_server(debug=False,port=8897)

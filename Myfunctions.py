#Author KARKOURI FATIMA-ZAHRA Git fzkarkouri
#Definitions of functions used in the Medical Cost Prediction Notebook

#Data manipulation
import pandas as pd
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option('display.max_colwidth', None)

import numpy as np

#Statistics
from scipy.stats import pointbiserialr  #To calculate the Point Biserial correlation coefficient
from scipy.stats import kruskal         #To perform the Kruskal–Wallis one-way analysis of variance
from scipy.stats import levene          #To perform the Levene Test
from scipy.stats import chi2_contingency,chi2 #To perform the chi2 test



#Data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures

#Feature Selection
from sklearn.inspection import permutation_importance

#Regression Algorithms
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, ElasticNet, ElasticNetCV, ARDRegression
 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

#Metrics
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score


#Plot the regression metrics results for a specified algorithm by number of columns
#The data used for the plots are from the metrics data frame we created in the prediction step
#The metrics we plot are : R2, MSE, MAE, ME
###############################################################################################
###############################################################################################
############################plot_errors########################################################
###############################################################################################
###############################################################################################
def plot_errors(metrics,
                algorithm = 'Linear Regression', #The algorithm for which we plot the metrics 
                r2_positive_only = False,        #keep the possibility to plot only the positive R2
                k_best_scores = None):           #The K best scores in the 4 metrics if not specified all the scores will be plotted 
    
    

    
                fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 6),squeeze=False)
                fig.tight_layout(pad=4.0)
                data = metrics[metrics['Algorithm']==algorithm]

                #Plot the R2
                if k_best_scores is None:
                    if r2_positive_only : 
                        data_r2 = data[data['R2 score']>0]
                    else:
                        data_r2 = data
                else:
                      data_r2 = data.nlargest(k_best_scores, 'R2 score')
                        
                sns.lineplot(data_r2['Number of columns'],
                             data_r2['R2 score'],
                             #label = 'R2',
                             ax=axes[0][0])
                
        
                #Mark the optimal point
                #print(data_r2.nlargest(1, 'R2 score')['Number of columns'].values)
                x = data_r2.nlargest(1, 'R2 score')['Number of columns'].values[0]
                y = data_r2.nlargest(1, 'R2 score')['R2 score'].values[0]
                
                axes[0][0].text(x = x, y = y, 
                                s = '({},R2={:.4f})'.format(x,y),
                                fontsize = 'large')

                axes[0][0].set_title('R-Squared', size=12)
                #axes[0][0].legend()

                #Plot the MSE
                if k_best_scores is None:
                      data_mse = data
                else:
                      data_mse = data.nsmallest(k_best_scores, 'Mean squared error')

                sns.lineplot(data_mse['Number of columns'],
                             data_mse['Mean squared error'],
                             #label = 'MSE',
                             ax=axes[0][1])

                 #Mark the optimal point
                x = data_mse.nsmallest(1, 'Mean squared error')['Number of columns'].values[0]
                y = data_mse.nsmallest(1, 'Mean squared error')['Mean squared error'].values[0]
                axes[0][1].text(x = x, y = y, 
                                s = '({},MSE={:.1f})'.format(x,y),
                                fontsize = 'large')

                axes[0][1].set_title('Mean squared error', size=12)
                #axes[0][1].legend()

                #Plot the MAE

                if k_best_scores is None:
                      data_mae = data
                else:
                      data_mae = data.nsmallest(k_best_scores, 'Mean absolute error')

                sns.lineplot(data_mae['Number of columns'],
                         data_mae['Mean absolute error'],
                         #label = 'MAE',
                         ax=axes[0][2])

                #Mark the optimal point
                x = data_mae.nsmallest(1, 'Mean absolute error')['Number of columns'].values[0]
                y = data_mae.nsmallest(1, 'Mean absolute error')['Mean absolute error'].values[0]
                axes[0][2].text(x = x, y = y, 
                                s = '({},MAE={:.1f})'.format(x,y),
                                fontsize = 'large')

                axes[0][2].set_title('Mean absolute error', size=12)
                #axes[0][2].legend()


                #Plot the ME

                if k_best_scores is None:
                      data_me = data
                else:
                      data_me = data.nsmallest(k_best_scores, 'Max error')

                sns.lineplot(data_me['Number of columns'],
                         data_me['Max error'],
                         #label = 'ME',
                         ax=axes[0][3])

                #Mark the optimal point
                x = data_me.nsmallest(1, 'Max error')['Number of columns'].values[0]
                y = data_me.nsmallest(1, 'Max error')['Max error'].values[0]
                axes[0][3].text(x = x, y = y, 
                                s = '({},ME={:.1f})'.format(x,y),
                                fontsize = 'large')

                axes[0][3].set_title('Max error', size=12)
                #axes[0][3].legend()

                fig.suptitle('{} Errors by number of columns'.format(algorithm), fontsize=16)
                plt.show()
                
###############################################################################################
###############################################################################################
############################plot_errors########################################################
###############################################################################################
###############################################################################################                

#Computes the scores : R2, MSE, MAE, ME for different number of columns
#The Feature Permutation Importance method is used to select the most import k features
#The results are then returned in a dataframe
###############################################################################################
###############################################################################################
############################permutation_importance_percolnumber################################
###############################################################################################
###############################################################################################
def permutation_importance_percolnumber(X ,
                                        y ,
                                        X_test ,
                                        y_test ,
                                        algorithm ):

                #Metrics data
                metrics_data = pd.DataFrame(columns = ['Algorithm','Number of columns','Columns','Max error','Mean absolute error','Mean squared error','R2 score'])

                #Estimator
                if algorithm in ['Linear Regression','Polynomial Regression']:
                    regressor = LinearRegression().fit(X, y)#normalize=True
                elif algorithm == 'Lasso Regression':
                    regressor = Lasso(max_iter=50000).fit(X, y)
                elif algorithm == 'Ridge Regression':
                    regressor = Ridge().fit(X, y)
                elif algorithm == 'ElasticNet Regression':
                    regressor = ElasticNet().fit(X, y)
                elif algorithm == 'K-Nearest Neighbors Regression':
                    regressor = KNeighborsRegressor().fit(X, y)
                elif algorithm == 'Support Vector Machine Regression':
                    regressor = SVR(kernel='poly',degree=2,coef0=0.5,C=100).fit(X, y)#,degree=2,C=50
                elif algorithm == 'Decision Tree Regression':
                    regressor = DecisionTreeRegressor().fit(X, y)
                elif algorithm == 'Random Forest Regression':
                    regressor = RandomForestRegressor().fit(X, y)
                elif algorithm == 'Gradient Boosting Regression':
                    regressor = GradientBoostingRegressor().fit(X, y)


                #Feature Selector
                selector = permutation_importance(regressor, X, y, n_repeats=5, scoring='r2', random_state=10)
                feature_importance = selector.importances_mean

                nbr_columns = X.shape[1]
                for i in range(5,nbr_columns+1):

                    try:
                        selected_Features = X.columns[np.argsort(feature_importance)[nbr_columns-i:]]
                        X_new = X[selected_Features]
                        X_test_new = X_test[selected_Features]
                    except:
                        selected_Features = np.argsort(feature_importance)[nbr_columns-i:]
                        X_new = X[:,selected_Features]
                        X_test_new = X_test[:,selected_Features]



                    #Training of the model

                    if algorithm in ['Linear Regression','Polynomial Regression']:
                        regressor = LinearRegression().fit(X_new, y)#normalize=True
                    elif algorithm == 'Lasso Regression':
                        regressor = Lasso(max_iter=50000).fit(X_new, y)
                    elif algorithm == 'Ridge Regression':
                        regressor = Ridge().fit(X_new, y)
                    elif algorithm == 'ElasticNet Regression':
                        regressor = ElasticNet().fit(X_new, y)
                    elif algorithm == 'K-Nearest Neighbors Regression':
                        regressor = KNeighborsRegressor().fit(X_new, y)
                    elif algorithm == 'Support Vector Machine Regression':
                        regressor = SVR(kernel='poly',degree=2,coef0=0.5,C=100).fit(X_new, y)#kernel='poly',degree=2,C=50
                    elif algorithm == 'Decision Tree Regression':
                        regressor = DecisionTreeRegressor().fit(X_new, y)
                    elif algorithm == 'Random Forest Regression':
                        regressor = RandomForestRegressor().fit(X_new, y)
                    elif algorithm == 'Gradient Boosting Regression':
                        regressor = GradientBoostingRegressor().fit(X_new, y)



                    #Prediction
                    y_pred =  regressor.predict(X_test_new)
                    #Metrics Computing
                    ME = max_error(y_test,y_pred)
                    MAE= mean_absolute_error(y_test,y_pred)
                    MSE= mean_squared_error(y_test,y_pred)
                    R2 = r2_score(y_test,y_pred)
                    #Append the Metrics Dataframe
                    metrics_data = metrics_data.append(pd.Series([algorithm,i,selected_Features, ME,MAE,MSE,R2],
                                                 index=metrics_data.columns),
                                                 ignore_index=True)
                return metrics_data
            
###############################################################################################
###############################################################################################
############################permutation_importance_percolnumber################################
###############################################################################################
###############################################################################################            
            
            
            
#Display the optimal row for each metric
#Return the optimal columns for R2
###############################################################################################
###############################################################################################
############################optimalColumns#####################################################
###############################################################################################
###############################################################################################
def optimalColumns(algorithm,metrics,columns,features_name=None): 
    
            selected_features = metrics[metrics['Algorithm']==algorithm].nlargest(1, 'R2 score')['Columns'].values[0].tolist()
            print('The Features Permutation Importance selected {} columns that optimised the R2 score.'.format(len(selected_features)))
            if (algorithm == 'Polynomial Regression'):
                selected_features_names = [features_name[i] for i in selected_features]
                selected_features_names = [f.split('x')[1:] for f in  selected_features_names]
                features = []
                for f in selected_features_names:
                    feature = [ ( '({})^2'.format(columns[int(i.split('^')[0])]) if ('^' in i) else ('{}'.format(columns[int(i.split('^')[0])]))) for i in f]
                    features.append(feature)
                    
                if len(features)<=25 :
                    print('The selected columns are : {}'.format(features))
                else:
                    print('The first 25 selected columns are : {}'.format(features[:25]))

            elif (isinstance(selected_features[0],int)):
                
                print('The selected columns are : {}'.format(columns[selected_features]))
            else:
                print('The selected columns are : {}'.format(selected_features))
            return selected_features
###############################################################################################
###############################################################################################
############################optimalColumns#####################################################
###############################################################################################
###############################################################################################

#Plot the graph Actual Vs Predicted for a giving regressor 
###############################################################################################
###############################################################################################
############################plot_ActualVsPredicted#############################################
###############################################################################################
###############################################################################################
def plot_ActualVsPredicted(regressor,X_test,y_test,algorithm):

    #Predict charges for the Test data set
    y_pred = regressor.predict(X_test)

   
    #Plot the graph
    plt.figure(figsize=(10, 8))
    plt.plot(y_test,y_pred,'r+')
    plt.plot(y_test,y_test)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('{}\'s Actual Vs Predicted'.format(algorithm))
    plt.show()
    
    #Print the regression metrics 
    print('R-squared : {}'.format(r2_score(y_test,y_pred)))
    print('Max Error : {}'.format(max_error(y_test,y_pred)))
    print('Mean Absolute Error : {}'.format(mean_absolute_error(y_test,y_pred)))
    print('Mean Squared Error : {}'.format(mean_squared_error(y_test,y_pred)))
###############################################################################################
###############################################################################################
############################optimalColumns#####################################################
###############################################################################################
###############################################################################################
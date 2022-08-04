import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import math


def read_file(filename_path):
    count_slash = filename_path.count('/')
    filename = filename_path.split('/', count_slash)[-1]  # filename is after the last slash
    foldername = filename_path.split('/', count_slash)[-2]  # folder where filename is at   
 
    # reads in dataset
    df = pd.read_csv('/cluster/ifs/projects/AlphaThal/MachineLearning/Features/' + foldername + '/' + filename, sep='\t')
    
    # randomly shuffles dataset
    df = df.sample(frac=1)
    
    # drops 'Genotype' column from original dataframe, saves 'x' variables
    df_x = df.drop(['Genotype'], axis=1)

    # represents actual mutated genotype for each observation, saves 'y' variables
    df_y = df['Genotype'] 

    # grabs coverage for each chromosomal section
    df_column_names = df.columns
    
    # stores the 6 mutation genotypes in a list
    mutated_genotype = df.Genotype.unique()

    return df_x, df_y, df_column_names, mutated_genotype


# shuffles data set
def shuffle(df_x, df_y):
    # takes 100 random samples out and stores it
    df_variables_test = df_x[:100]
    df_output_genotype_test = df_y[:100]

    # keeps rest of variables not taken out to train model
    df_variables_train = df_x[100:]
    df_output_genotype_train = df_y[100:]
    
    return df_variables_train, df_variables_test, df_output_genotype_train, df_output_genotype_test


# finds best parameters using GridSearchCV
def grid(df_x, df_y):
    limit = int(math.log2(len(df_x.index)))

    n_trees = [100, 200, 300, 400, 500]
    max_dep = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    if (limit in max_dep) == False:
        for m in max_dep:
            if limit < m:
                max_dep.insert(max_dep.index(m), limit)
                break

    if max_dep[-1] < limit:
        max_dep.append(limit)
    
    
    parameters = {'n_estimators':n_trees, 'max_depth':max_dep}
    
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=2), param_grid=parameters,
                          return_train_score=True, cv=5)
    rf_grid.fit(df_x, df_y)
    
    return rf_grid


# calculates cross validation score of model with default hyper parameters
# ALSO returns model created. BUT can also just use .best_estimator from GridSearchCV
# tutorial referenced:
### https://www.youtube.com/watch?v=gJo0uNL-5Qw
def calculate_score(df, output, best_n_trees, best_max_depth):

    model = RandomForestClassifier(n_estimators=best_n_trees, max_depth=best_max_depth,
                                       random_state=2)
    model.fit(df, output)

    cross_score = cross_val_score(model, df, output, cv=5)
    
    return model, cross_score


# reads in simulated dataset that will be used to create model for real data
read_file_sim = read_file('/cluster/ifs/projects/AlphaThal/MachineLearning/Features/DataSet2/DataSet2.11760samples.cvg30.insert440.stdev100.hg19.txt')
shuffle_sim = shuffle(read_file_sim[0], read_file_sim[1])
grid_sim = grid(shuffle_sim[0], shuffle_sim[2])

# testing simulated model with 100 test points that were taken out beforehand
test_preds = cal_score_output_sim[0].predict(shuffle_sim[1])  # predicting x variables from test set using model
test_actual = shuffle_sim[3]
print(accuracy_score(test_actual, test_preds))


# reads in file with real data
# NO shuffling
def read_real_data(filename):

    # reads in dataframe that has real data
    df = pd.read_csv(filename, sep='\t')
    
    # drops 'Genotype' column from original dataframe, saves 'x' variables
    df_x = df.drop(['Genotype'], 1)

    # represents actual mutated genotype for each observation, saves 'y' variables
    df_y = df['Genotype'] 

    return df_x, df_y


best_params_sim = grid_sim.best_params_  # finds best parameters
cal_score_output_sim = calculate_score(shuffle_sim[0].to_numpy(), shuffle_sim[2].to_numpy(), best_params_sim['n_estimators'], best_params_sim['max_depth'])

model_sim = cal_score_output_sim[0]  # model from simulated data
model_grid =  grid_sim.best_estimator_  # model from grid


# reads in real data
read_file_real = read_real_data('/cluster/ifs/projects/AlphaThal/pcr_results/RealSamples.unnormalized.nogenotypes.hg19.test.norm.maskedgenos.final.txt')

y_pred_sim = model_sim.predict(read_file_real[0].to_numpy())  # predicts output from real data using model from 'calculate_score' func
y_pred_grid = model_grid.predict(read_file_real[0].to_numpy())  # finds genotype probability for predictions using model

y_prob_sim = model_sim.predict_proba(read_file_real[0].to_numpy()) # predicts output from real data using .best_estimator_ from GridSearchCV
y_prob_grid = model_grid.predict_proba(read_file_real[0].to_numpy()) # finds genotype probability for predictions using GridSearchCV


# creates textfiles to put all predictions/probability predictions in 
with open('/cluster/ifs/projects/AlphaThal/sandboxes/jennifer/RandomForest_AlphaThal/genotype_output_grid.txt', 'w') as writer:
  for index, genotype in enumerate(y_pred_grid):
    writer.write(str(index+1) + ': ' + genotype + '\n')

with open('/cluster/ifs/projects/AlphaThal/sandboxes/jennifer/RandomForest_AlphaThal/genotype_output_grid_probs.txt', 'w') as writer:
  for i in range(len(y_prob_grid)):
    output = y_prob_grid[i]
    writer.write(str(output) + '\n')

with open('/cluster/ifs/projects/AlphaThal/sandboxes/jennifer/RandomForest_AlphaThal/genotype_output.txt', 'w') as writer:
  for index2, genotype2 in enumerate(y_pred_sim):
    writer.write(str(index2+1) + ': ' + genotype2 + '\n')

with open('/cluster/ifs/projects/AlphaThal/sandboxes/jennifer/RandomForest_AlphaThal/genotype_output_probs.txt', 'w') as writer:
  for i in range(len(y_prob_sim)):
    output = y_prob_sim[i]
    writer.write(str(output) + '\n')


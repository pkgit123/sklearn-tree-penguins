#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Example (Penguins)
# 
# ### Code to convert notebook into Python code
# ```shell
# jupyter nbconvert --to script *.ipynb
# ```
# ### Dataset References: 
# * Penguin dataset.  https://www.kaggle.com/parulpandey/penguin-dataset-the-new-iris
# * Seaborn penguins dataset.  https://seaborn.pydata.org/introduction.html
# * Kaggle IBM-HR notebooks.  https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/code?datasetId=1067&sortBy=voteCount
# * Kaggle IBM-HR Decision Tree.  https://www.kaggle.com/hackspyder/decision-based-approach
# * Kaggle IBM-HR Logistic Regression: https://www.kaggle.com/faressayah/ibm-hr-analytics-employee-attrition-performance
# 
# ### Decision Tree References: 
# * Scikit-Learn Decision Tree.  https://stackoverflow.com/questions/38108832/passing-categorical-data-to-sklearn-decision-tree
# * https://www.kaggle.com/hackspyder/decision-based-approach
# * https://stackoverflow.com/questions/5316206/converting-dot-to-png-in-python
# * https://github.com/pydot/pydot
# * https://stackoverflow.com/questions/27817994/visualizing-decision-tree-in-scikit-learn
# * https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
# * https://mljar.com/blog/visualize-decision-tree/

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 999)

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print('Seaborn version: ', sns.__version__)


# In[2]:


# https://www.kaggle.com/hackspyder/decision-based-approach

from sklearn import tree

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.linear_model import LogisticRegression

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.model_selection import train_test_split


# In[3]:


import os


# In[ ]:





# In[4]:


# load penguins dataset
df_penguins_raw = sns.load_dataset("penguins")

print('Table shape: ', df_penguins_raw.shape, '\n')
print('Number of Uniques per column: ')
print(df_penguins_raw.nunique(), '\n')
print('Number of nulls per column: ')
print(df_penguins_raw.isna().sum(), '\n')
print('Dataframe info: ')
print(df_penguins_raw.info(), '\n')

df_penguins_raw.head()


# In[5]:


# make copy for cleaning
df_penguins = df_penguins_raw.copy()


# In[6]:


# Clean up null values: .dropna() of target variable
df_penguins = df_penguins.dropna(subset=['sex'])


# In[7]:


# Clean up null values: .fillna() of numeric variable using .mean()
df_penguins['bill_length_mm'].fillna(df_penguins['bill_length_mm'].mean(), inplace=True)
df_penguins['bill_depth_mm'].fillna(df_penguins['bill_depth_mm'].mean(), inplace=True)
df_penguins['flipper_length_mm'].fillna(df_penguins['flipper_length_mm'].mean(), inplace=True)
df_penguins['body_mass_g'].fillna(df_penguins['body_mass_g'].mean(), inplace=True)


# In[8]:


# add categorical bin for numeric
df_penguins[f'group_body_mass'] = pd.cut(df_penguins['body_mass_g'], bins=3)


# In[ ]:





# ### Exploratory Data Analysis
# * Pivot tables
# * Grouped bar charts
# * Scatterplots
# * Histograms

# In[9]:


# pivot table to count categorical columns: island vs. gender
# values='species' is any column without nulls
# index='island' is column to put in rows
# margins=True means to include totals
# .sort_values('All') means sort by index totals
pt_island_gender = pd.pivot_table(df_penguins, values='species', index='island', columns='sex', aggfunc=len, margins=True).fillna(0).sort_values('All', ascending=False)
pt_island_gender


# In[10]:


# pivot table to count categorical columns: species vs. gender
pt_island_gender = pd.pivot_table(df_penguins, values='island', index='species', columns='sex', aggfunc=len, margins=True).fillna(0).sort_values('All', ascending=False)
pt_island_gender


# In[11]:


get_ipython().run_cell_magic('time', '', '\nx=\'bill_length_mm\'\ny=\'bill_depth_mm\'\nhue=\'species\'\ndata=df_penguins\n\nsns.scatterplot(data=data, x=x, y=y, hue=hue, x_jitter=True)\n\n# put a title\nplt.title(f"`{x}` vs. `{y}` with hue of `{hue}`")\n\n# Rotates X-Axis Ticks by 90-degrees\nplt.xticks(rotation = 90) \n\n# Put the legend out of the figure\nplt.legend(title=f"Legend: {hue}", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)\n\nplt.show()\nprint()')


# In[12]:


get_ipython().run_cell_magic('time', '', '\nx=\'bill_length_mm\'\ny=\'bill_depth_mm\'\nhue=\'island\'\ndata=df_penguins\n\nsns.scatterplot(data=data, x=x, y=y, hue=hue, x_jitter=True)\n\n# put a title\nplt.title(f"`{x}` vs. `{y}` with hue of `{hue}`")\n\n# Rotates X-Axis Ticks by 90-degrees\nplt.xticks(rotation = 90) \n\n# Put the legend out of the figure\nplt.legend(title=f"Legend: {hue}", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)\n\nplt.show()\nprint()')


# In[13]:


get_ipython().run_cell_magic('time', '', '\nx=\'bill_length_mm\'\ny=\'bill_depth_mm\'\nhue=\'sex\'\ndata=df_penguins\n\nsns.scatterplot(data=data, x=x, y=y, hue=hue, x_jitter=True)\n\n# put a title\nplt.title(f"`{x}` vs. `{y}` with hue of `{hue}`")\n\n# Rotates X-Axis Ticks by 90-degrees\nplt.xticks(rotation = 90) \n\n# Put the legend out of the figure\nplt.legend(title=f"Legend: {hue}", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)\n\nplt.show()\nprint()')


# In[14]:


get_ipython().run_cell_magic('time', '', '\nx=\'bill_length_mm\'\ny=\'bill_depth_mm\'\nhue=\'group_body_mass\'\ndata=df_penguins\n\nsns.scatterplot(data=data, x=x, y=y, hue=hue, x_jitter=True)\n\n# put a title\nplt.title(f"`{x}` vs. `{y}` with hue of `{hue}`")\n\n# Rotates X-Axis Ticks by 90-degrees\nplt.xticks(rotation = 90) \n\n# Put the legend out of the figure\nplt.legend(title=f"Legend: {hue}", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)\n\nplt.show()\nprint()')


# In[15]:


def plot_gb_pct_total(gb_col_name, str_col_name, df_input):
    '''
    Create groupby % of total based on two categorical variables.
    
    Input examples:
        gb_col_name = 'SUB'
        str_col_name = 'CATEGORY'
    '''
    
    dfu = df_input.groupby([gb_col_name])[str_col_name].value_counts(dropna=False, normalize=True).unstack().T
    dfu = dfu.sort_values(dfu.columns[0], ascending=False)

    # plot
    dfu.plot.bar(figsize=(7, 5))
    # ax = sns.barplot(x=gb_col_name, y=str_col_name, data=dfu)
    plt.title(f"Grouped barchart: % of Total by `{str_col_name}`, groupby `{gb_col_name}`")
    plt.legend(title=gb_col_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel(str_col_name)
    plt.ylabel('% of Total')
    plt.show()
    
    print('\n')
    print(dfu)


# In[16]:


# Check balanced categories: gender vs. island
plot_gb_pct_total(
    gb_col_name = 'sex', 
    str_col_name = 'island', 
    df_input = df_penguins
)


# In[17]:


# Check balanced categories: species vs. island
# species `Adelie` is balanced across islands
# species `Chinstrap` is unbalanced, only in island `Dream`
# species `Gentoo` is unbalanced, only in island `Biscoe`
plot_gb_pct_total(
    gb_col_name = 'species', 
    str_col_name = 'island', 
    df_input = df_penguins
)


# In[18]:


# bill_length_mm	bill_depth_mm	flipper_length_mm	body_mass_g


# In[19]:


x="bill_length_mm"
sns.histplot(data=df_penguins, x=x)

# put a title
plt.title(f"Histogram of `{x}`")

plt.show()


# In[20]:


x="bill_depth_mm"
sns.histplot(data=df_penguins, x=x)

# put a title
plt.title(f"Histogram of `{x}`")

plt.show()


# In[21]:


x="flipper_length_mm"
sns.histplot(data=df_penguins, x=x)

# put a title
plt.title(f"Histogram of `{x}`")

plt.show()


# In[22]:


x="body_mass_g"
sns.histplot(data=df_penguins, x=x)

# put a title
plt.title(f"Histogram of `{x}`")

plt.show()


# In[ ]:





# ### Encode Columns (Categorical vs. Numeric)

# In[23]:


def encode_columns_split(
    in_df, 
    in_ls_cols_numeric, 
    in_str_target_variable,
    in_ls_cols_drop=['blank_drop_xyz'], 
    drop_cols=True
):
    '''
    Split the dataset column-wise into 4 subsets:
        * Categorical columns
        * Numeric columns
        * Target variable
        * Drop columns
        
    Inputs:
        * in_df - dataframe, input data
        * in_ls_cols_numeric - list, columns with numeric data
        * in_str_target_variable - list, columns with categorical data
        * in_ls_cols_drop - list, columns to drop from dataset
        * drop_cols=True
    Return:
        * temp_df_final - dataframe, merge numeric columns and categorical columns (one-hot-encoding)
        * target_var_output - series, output column from dataframe
        * target_output_categories - series, unique alphabetical output categories
    '''
    
    print("Input dataframe shape: ", in_df.shape)
    print("in_ls_cols_numeric: ", in_ls_cols_numeric)
    print("in_str_target_variable: ", in_str_target_variable)
    print("in_ls_cols_drop: ", in_ls_cols_drop)
    print()
    
    
    # ============================================================
    # create list of categoric, numeric columns
    # ============================================================
    
    # create list with numeric and target variable
    temp_ls_cols_numeric_target = in_ls_cols_numeric+[in_str_target_variable]
    print("temp_ls_cols_numeric_target", temp_ls_cols_numeric_target)
    print()
    
    
    # create list of categorical columns (exclude the numeric columns and the target variable)
    temp_ls_cols_cat = in_df.columns.difference(temp_ls_cols_numeric_target)
    print("temp_ls_cols_cat: ", temp_ls_cols_cat)
    print()
    
    # ============================================================
    # create output vector (target variable), and unique list of output categories
    # ============================================================
    
    # create output vector (target variable)
    target_var_output = in_df[in_str_target_variable]
    print("Length of target_var_output: ", len(target_var_output))
    print()
    
    # create list of output categories
    target_output_categories = sorted(in_df[in_str_target_variable].value_counts().index)
    print("target_output_categories: ", target_output_categories)
    print()
    
    # ============================================================
    # use this dataframe for remaining code, rather than input df
    # ============================================================
    
    # drop the target variable from the training data
    temp_df_drop = in_df.drop([in_str_target_variable], axis=1)
    print("Shape temp_df_drop: ", temp_df_drop.shape)
    
    # create numeric dataframe
    temp_df_num = temp_df_drop[in_ls_cols_numeric]
    print("Shape temp_df_num: ", temp_df_num.shape)
    
    # create categoric dataframe 
    temp_df_cat_raw = temp_df_drop[temp_ls_cols_cat]
    print("Shape temp_df_cat_raw: ", temp_df_cat_raw.shape)
    
    # modify categoric dataframe to drop some columns
    temp_ls_cat_drop = [x for x in temp_df_cat_raw.columns if x not in in_ls_cols_drop]
    temp_df_cat_drop = temp_df_cat_raw[temp_ls_cat_drop]
    print("Shape temp_df_cat_drop: ", temp_df_cat_drop.shape)
        
    # convert the categorical dataframe, encoded as numeric, using "one-hot-encoding"
    temp_df_cat = pd.get_dummies(temp_df_cat_drop)
    print("Shape temp_df_cat: ", temp_df_cat.shape)
    
    # Concat the two dataframes together columnwise
    temp_df_final = pd.concat([temp_df_num, temp_df_cat], axis=1)
    print("Shape temp_df_final: ", temp_df_final.shape)
    
    return temp_df_final, target_var_output, target_output_categories


# In[24]:


df_penguins.head()


# In[25]:


# ===================================================
# Prepare for model: Total dataset
# ===================================================

# create list of numeric columns
ls_cols_numeric = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
]

# prepare the data for the decision tree model
df_final_out, ls_target_train_out, ls_target_categories_out = encode_columns_split(
    df_penguins, 
    in_ls_cols_numeric=ls_cols_numeric,
    in_str_target_variable='sex',
    in_ls_cols_drop=['group_body_mass'],
    drop_cols=True
)


# In[26]:


def build_decision_tree(train, target_train, tree_feature_names, tree_class_names, str_filename_output):
    '''
    Prepare the data input and model parameters to the model.  Build decision tree.
    
    Inputs:
        * train - dataframe, no need to split dataset for decision tree, just use everything
        * target_train - list, single dataframe column of output or target variable
        * tree_feature_names - list, columns from dataframe
        * tree_class_names - sorted list of output categories
        * str_filename_output - str, prefix for filename output (e.g. dot-file, png-file)
    Model hyper-parameters, hard-coded:
        * tree_max_depth = 3
        * tree_filled = True
        * tree_impurity = False
        * tree_node_ids = True
        * tree_proportion = False
        * tree_rounded = True
        * tree_fontsize = 12
    Outputs, create local files:
        * dot-file
        * png-file
    '''
    
    # =============================================
    # Model input parameters
    # =============================================
    
    # use the "df_final" dataframe ... has been encoded (numeric), removed nulls, removed target variable
    # for now, put everything in a training dataset, ignore the train/test split dataset
    # this is like capital "X"
    train = train    # this is like capital "X"
    
    # use the "emp_status" column from the original dataset as the target variable
    target_train = target_train    # this is like lowercase "y"
    
    # use the column names from the "df_final" dataframe for list of features
    tree_feature_names = tree_feature_names
    
    # use the values in the target variable for class_names ... this list needs to be sorted ascending
    tree_class_names = sorted(tree_class_names)
    
    # =============================================
    # Model hyper-parameters
    # =============================================
    
    # store other model variables
    tree_max_depth = 3
    tree_filled = True
    tree_impurity = False
    tree_node_ids = True
    tree_proportion = False
    tree_rounded = True
    tree_fontsize = 12
    
    print('Total number of features used in model: ', len(tree_feature_names))
    print('First 5 feature names: ', tree_feature_names[:5])
    print('Class names: ', tree_class_names)
    
    # =============================================
    # Model training
    # =============================================
    
    # instantiate and fit the Decision Tree Classifier
    decision_tree = tree.DecisionTreeClassifier(max_depth = tree_max_depth)
    decision_tree.fit(train, target_train)
    
    # =============================================
    # Model output filenames and paths
    # =============================================
    
    # build string for output filenames: .dot and .png
    # include the tree depth as part of filename
    # include today's date as part of the filename
    str_date_today = str(pd.Timestamp.now().date())
    str_filename_output = str_filename_output
    str_filename_dotfile = f"{str_date_today}_{str_filename_output}_depth{tree_max_depth}.dot"
    str_filename_pngfile = f"{str_date_today}_{str_filename_output}_depth{tree_max_depth}.png"

    # path folder and filename
    str_folder_decisiontree = 'output-decision-trees'
    if not os.path.exists(str_folder_decisiontree):
        os.makedirs(str_folder_decisiontree)
        
    # create path to .dot and .png files
    str_pathfile_dotfile = os.path.join(str_folder_decisiontree, str_filename_dotfile)
    str_pathfile_pngfile = os.path.join(str_folder_decisiontree, str_filename_pngfile)
    
    print(str_pathfile_dotfile)
    print(str_pathfile_pngfile)
    
    
    
    # =============================================
    # output dot-file
    # =============================================
    
    # Export our trained model as a .dot file
    with open(str_pathfile_dotfile, 'w') as f:
        f = tree.export_graphviz(
            decision_tree,
            out_file=f,
            max_depth = tree_max_depth,
            impurity = tree_impurity,
            feature_names = tree_feature_names,
            class_names = tree_class_names,
            rounded = tree_rounded,
            filled= tree_filled 
        )
        
    # =============================================
    # output png-file
    # =============================================

    plt.figure(figsize=(40,20))  # customize according to the size of your tree

    _ = tree.plot_tree(
        decision_tree, 
        max_depth = tree_max_depth,
        feature_names = tree_feature_names,
        class_names = tree_class_names,
        filled = tree_filled,
        impurity = tree_impurity,
        proportion = tree_proportion,
        node_ids = tree_node_ids,
        rounded = tree_rounded,
        fontsize = tree_fontsize
    )

    # saving the file.Make sure you 
    # use savefig() before show().
    plt.savefig(str_pathfile_pngfile)

    plt.show()


# In[27]:


# ===================================================
# Build decision tree model: total dataset
# ===================================================

build_decision_tree(
    train = df_final_out,
    target_train = ls_target_train_out, 
    tree_feature_names = df_final_out.columns.values,
    tree_class_names = ls_target_categories_out,
    str_filename_output = "tree1"
)


# In[ ]:





"""
Katheryn Lanman and Teera Tesharojanasup

DS2500 Project 2 Code

Data sourced from:
    
    https://nces.ed.gov/ipeds/search?query=&query2=&resultType=all&page=1&sortBy=date_desc&overlayTableId=25002 

    https://fred.stlouisfed.org/series/FPCPITOTLZGUSA
    
    https://www.statista.com/statistics/237681/international-students-in-the-us/
    
    https://datausa.io/profile/university/northeastern-university#costs
    
    https://educationdata.org/average-student-loan-debt-by-year

"""

import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

FILES = ['IPEDS1', 'IPEDS2', 'IPEDS3', 'IPEDS4', 'IPEDS5']

def read_csv(filename):
      '''
      read csv file into pandas dataframe
    
      Parameters
      ---
      filename: string. name of file to read
    
      Returns
      ---
      data: dataframe containing file data
      '''
      data = pd.read_csv(filename, header = None)
      
      return data

def drop_rows(df, keyword, drop_key = True):
      '''
      drop all rows above row containing a certain keyword. drop row containing 
      keyword if drop_key = True.
    
      Parameters
      ---
      df: dataframe to manipulate
      keyword: string. keyword to search for
      drop_key: boolean. if true, drop row containing keyword
    
      Returns
      ---
      drop_list: list of indices of rows to drop
      '''
      drop_list = []
      for index, row in df.iterrows():
          if keyword not in row.values:
            drop_list.append(index)
          else:
            if drop_key:
              drop_list.append(index)
            break
        
      return drop_list

def replace_row(df, vals, loc, rows):
      '''
      replaces row/rows of dataframe with new row at given location
    
      Parameters
      ---
      df: dataframe to manipulate
      vals: list. values for replacement row
      loc: numeric. index location to place replacement row.
      rows: ints. indices of rows to replace
    
      Returns
      ---
      df: dataframe with old rows replaced
      '''
      df.loc[loc] = vals
      df = df.drop(rows)
      df = df.sort_index().reset_index(drop = True)
      
      return df

def merge_dflist(df_list):
      '''
      merges a list of dataframes into one. by index
    
      Parameters
      ---
      df_list: list of dataframes
    
      Returns
      ---
      merged: new merged dataframe
      '''
      merged = df_list[0]
      for i in range(1, len(df_list)):
          merged = merged.merge(df_list[i], left_index = True, right_index = True)
        
      return merged

def remove_punc(df):
      '''
      removes punctuation/special chars (i.e. '$', '-', '.', etc.)
      from all cells of a dataframe
    
      Parameters
      ---
      df: dataframe to manipulate
    
      Returns
      ---
      df: same dataframe but without punctuation
      '''
      df = df.applymap(lambda x: str(x).translate(str.maketrans('', '', 
                                 string.punctuation)))
      
      return df

def make_numeric(df, label):
      '''
      changes type of all desired columns to float. does not change the category
      column.
    
      Parameters
      ---
      df: dataframe to manipulate
      label: string. name of column containing category labels
    
      Returns
      ---
      df: same dataframe but with all desired cells as floats.
      '''
      for col in df:
          if col == label:
              continue
          df[col] = pd.to_numeric(df[col], errors='coerce')
        
      return df

def make_ave_row(cats, df, label, name):
      '''
      creates new row based on the average of certain other cells
    
      Parameters
      ---
      cats: list of strings. categories to average from
      df: dataframe to manipulate
      label: string. name of column containing labels/categories
      name: string. name for the new row
    
      Returns
      ---
      aves: list containing new row values
      '''
      subset = df[df[label].isin(cats)]
      aves = [name]
      for col in subset:
          if col == label:
              continue
          ave = round(subset[col].mean(), 0)
          aves.append(ave)
            
      return aves

def replace(df, label, keyword):
      '''
      searches for specified rows and shifts the data from any give one of these
      rows into the cells of the row directly above it.
      
      Parameters
      ---
      df: dataframe to manipulate
      label: string. name of column containing labels/categories
      keyword: word in column to search for
    
      Returns
      ---
      df: dataframe with specified changes
      '''
      index = 0
      while index < df.shape[0]:
          if df.iloc[index][label] == keyword:
              new_row = df.iloc[index].tolist()
              new_row[0] = df.iloc[index - 1][0]
              df = replace_row(df, new_row, ((2 * index) - 1) / 2, [index - 1, index])
          else:
              index += 1
          
      return df
  
def remove_last_digit(x):
    
    """
    gets rid of the last digit in a number
    
    Parameters
    ---
    x: an integer
    
    Returns
    ---
    an integer with its last digit removed
    
    Ex. Input  -> x = 10053
        Output -> new_x = 1005
    """
    
    return int(x / 10)    
  
def plot_scatter1(df, cat_name, target_cat, title_name, x_label, y_label):
    
    """
    plot a scatter plot with three lines of best fit
    
    Parameters
    ---
    df: dataframe containing info we want to use to plot
    cat_name: column we want to look into 
    target_cat: which row within the cat_name column we want the pull out data from
    
    Returns
    ---
    a scatter plot with three lines of best fit of the target_cat
    """    
    sns.set_theme()
    scatter_index = [] 
        
    for index, row in df.iterrows():
        if row[cat_name] == target_cat:
            scatter_index.append(index+1)
            scatter_index.append(index+2)
            scatter_index.append(index+3)
                  
    for i in range(len(scatter_index)):
        data = df.iloc[scatter_index[i]]
                    
        x = list(data.keys())
        y = list(data.values)
                        
        x.pop(0)
        label = y.pop(0)
                
        x_ints = [int(i[:4]) for i in x]
        x.insert(0, None)
                
        g = sns.regplot(x_ints, y, label=label)
        plt.legend()
    
    g.set_xticklabels(x)
    plt.xticks(fontsize = 12, rotation = 270)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_name)
    
def plot_scatter2(df1, df2, cat_name1, target_cat1, cat_name2, xlabel, ylabel, title_name, rotate):
    
    """
    plot a scatter plot using data from two separate dataframes

    Parameters
    ---
    df1: first dataframe
    df2: second dataframe
    cat_name1: the column name of the column we want to pull data from in df1
    target_cat1: the row name of the row we want to pull data from in df1
    cat_name2: the column name of the column we want to pull data from in df2
    xlabel: x-axis label
    ylabel: y-axis label
    title_name: title name
    rotate: x-ticks rotation 

    Returns
    ---
    a scatter plot of the points we want from df1 and df2 and the correlation
    coefficient of the scatter plot
    """
    
    sns.set_theme()
    scatter_index = [] 
        
    for index, row in df1.iterrows():
        if row[cat_name1] == target_cat1:
            scatter_index.append(index+1)
            scatter_index.append(index+2)
            scatter_index.append(index+3)
                  
    for i in range(len(scatter_index)):
        data = df1.iloc[scatter_index[i]]
                    
        x = list(df2[cat_name2])
        y = list(data.values)
        
        label = y.pop(0)
        analysis = stats.pearsonr(x, y)
        cc = analysis[0]
        
        sns.regplot(x, y, label=label + ": r = " + str(round(cc, 3)))
        plt.legend()

    plt.xticks(fontsize = 12, rotation = rotate)
    plt.ticklabel_format(style='plain', axis='x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_name)

def plot_scatter3(df, x_label, y_label, lgd_label):

    """
    plot a scatter plot with one line of best fit
    
    Parameters
    ---
    df: dataframe containing info we want to use to plot
    xlabel: x-axis label
    ylabel: y-axis label
    legend_label: legend label
    
    Returns
    ---
    a scatter plot with one line of best fit 
    """    
    
    x = list(df[x_label])
    y = list(df[y_label])
    
    sns.regplot(x, y, label=lgd_label)
    plt.legend()
    
    return x, y
    
def get_year(date):
    '''
    gets year out of date string.

    Parameters
    ---
    date: string. date in form YYYY-MM-DD

    Returns
    ---
    year: year from date string
    '''
    year = date.split('-')[0]
  
    return year

def main():
    # to store ipeds costs data sets
    df_list = []
    
    for file in FILES:
        df = read_csv(file + '.csv')
    
        # remove unwanted rows/columns. basic cleaning
        df = df.drop(drop_rows(df, '4-year'))
        df = df.drop(df.columns[[1, 4, 5, 6, 7, 8, 9, 10, 11, 12]], axis = 1)
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        df = df.rename(columns={df.columns[0]: 'School Category'})
        df = df.reset_index(drop = True)
        df = df.drop(drop_rows(df, 'Tuition and required fees', drop_key = False))
        df_list.append(df[:])
    
    # merge data for all years
    all_years = merge_dflist(df_list)
    
    # drop duplicate category columns
    all_years = all_years.drop(all_years.columns[[3, 5, 6, 9, 15, 16]], axis = 1)
    all_years = all_years.rename(columns={'2017–18_y': '2017–18'})
    
    # sort columns in order of academic year
    all_years = all_years.reindex(sorted(all_years.columns), axis=1)
    
    # move school category column to front
    cols = all_years.columns.tolist()
    all_years = all_years[[cols[-1]] + cols[:len(cols) - 1]]
    
    # drop duplicate year columns
    unique_cols = []
    for column in all_years:
        # deal with mismatched formatting
        if '–' in column:
            first_year = column.split('–')[0]
        else:
            first_year = column.split('-')[0]
        if first_year in unique_cols:
            all_years = all_years.drop(column, axis = 1)
        else:
            unique_cols.append(first_year)
        
    # drop unnecessary rows
    all_years.drop(all_years.tail(5).index, inplace = True)
    
    all_years = remove_punc(all_years)

    # remove unnecessary digits in school category column
    all_years['School Category'] = all_years.apply(lambda r:''.join([i for i in 
                                                   r['School Category'] if not 
                                                   i.isdigit()]), axis = 1)
    
    all_years = make_numeric(all_years, 'School Category')
    
    # condense public school tuition to be the average of in-district, in-state, 
    # and out-of-state tuitions
    public_cats = ['Indistrict', 'Instate', 'Outofstate']
    pub_aves = make_ave_row(public_cats, all_years, 'School Category', 'Public')
    all_years.reset_index(drop = True)
    all_years = replace_row(all_years, pub_aves, 2.5, [2, 3, 4, 5])
    
    # drop off-campus data
    all_years = all_years.drop([13, 16, 19, 24, 25, 28, 29, 32, 33])
    all_years = all_years.reset_index(drop = True)
    
    # want school categories to only use on campus data
    all_years = replace(all_years, 'School Category', 'On campus')
    
    # get northeastern tuition data
    tuitions = pd.read_csv('Tuition_Costs.csv')
    neu = tuitions[tuitions['University'] == 'Northeastern University']
    neu = neu.drop(['ID Year', 'ID University', 'Slug University'], axis = 1).reset_index(drop = True)
    neu = neu.drop([6])
    neu = neu.sort_values('State Tuition', ascending=True)
            
    # plot the scatter with different target categories    
    plot_scatter1(all_years, 'School Category', 'Tuition and required fees', 
                  'Cost of Tuition from 2014 to 2021', 'Year','Cost ($)') 
    plot_scatter3(neu, 'Year', 'State Tuition', 'Northeastern')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()

    plot_scatter1(all_years, 'School Category', 'Books and supplies', 
                  'Cost of Books and Supplies from 2014 to 2021', 'Year','Cost ($)')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()

    plot_scatter1(all_years, 'School Category', 'Room and board', 
                  'Cost of Room and Board from 2014 to 2021', 'Year','Cost ($)')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()

    plot_scatter1(all_years, 'School Category', 'Other expenses', 
                  'Cost of Other Expenses from 2014 to 2021', 'Year','Cost ($)')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
            
    # read in and clean international students data
    int_students = read_csv('international_students.csv')
    int_students = int_students.drop([0, 1, 2]).reset_index(drop = True)
    
    int_students = int_students.rename(columns={int_students.columns[0]: 'Year', 
                                int_students.columns[1]: 'International Students'})
    
    rows_to_drop = drop_rows(int_students, '2013/14')
    int_students = int_students.drop(rows_to_drop)
    
    int_students['International Students'] = int_students.apply(lambda r: 
                        r['International Students'].translate(str.maketrans('', '', 
                        string.punctuation)), axis = 1)
    int_students = make_numeric(int_students, 'Year')
    int_students['Year'] = int_students.apply(lambda r: r['Year'].split('/')[0], axis = 1)
                    
    # read in and clean US inflation data
    inflation = read_csv('us_inflation.csv')
    new_header = inflation.iloc[0]
    inflation = inflation[1:]
    inflation.columns = new_header
    inflation = inflation.rename(columns={inflation.columns[1]: 'Inflation Rate'}) \
                .reset_index(drop = True)
    inflation['DATE'] = inflation.apply(lambda r: get_year(r['DATE']), axis = 1)
    inflation['DATE'] = pd.to_datetime(inflation['DATE'])
    inflation['Year'] = pd.DatetimeIndex(inflation['DATE']).year
    inflation = inflation.drop('DATE', axis=1).set_index('Year').reset_index()
    inflation['Inflation Rate'] = inflation['Inflation Rate'].astype(float)
    inflation = inflation.loc[inflation['Year'] > 2013]
    
    # student loan debt data
    loan_debt = pd.read_csv('student_loan_debt.csv')
    loan_debt = loan_debt[loan_debt['Graduation Year'] > 2013].reset_index(drop = True)
    loan_debt = remove_punc(loan_debt)
    loan_debt = make_numeric(loan_debt, None)
    loan_debt['Graduation Year'] = loan_debt['Graduation Year'].apply(remove_last_digit)
    loan_debt = loan_debt.iloc[1:, :]
    loan_debt = loan_debt.sort_values('Graduation Year', ascending=True)

    # plot to see correlation between international students and tuition cost
    plot_scatter2(all_years, int_students, 'School Category', 
                  'Tuition and required fees', 'International Students', 
                  '# of International Students', 'Tuition Cost ($)',
                  'Tuition Cost Vs # of International Students (2014 - 2021)', 
                  270)   
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    
    # plot to see correlation between inflation rate and tuition cost
    plot_scatter2(all_years, inflation, 'School Category', 
                  'Tuition and required fees', \
                  'Inflation Rate', 'Inflation Rate', 'Tuition Cost ($)', 
                  'Tuition Cost Vs Inflation Rate (2014 - 2021)', 
                  0)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    
    # plot to see correlation between average student debt and tuition cost
    plot_scatter2(all_years, loan_debt, 'School Category', 
                  'Tuition and required fees',
                  'Debt at Graduation per Student', 
                  'Average Debt at Graduation of Students ($)',
                  'Tuition Cost ($)', 
                  'Tuition Cost Vs Average Student Debt (2014 - 2021)',
                  270)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
      
main()
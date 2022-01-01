import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
# print(df.head())


# Add 'overweight' column
df['bmi'] = (df['weight']/(df['height']/100)**2)
df['overweight'] = 0
df.loc[df['bmi'] > 25, ['overweight']] = 1
df.loc[df['bmi'] <= 25, ['overweight']] = 0

# print(df.head(20))
# print(df['overweight'].value_counts())

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

# print(df[['cholesterol', 'gluc']].describe())
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
# print(df[['cholesterol', 'gluc']].describe())


# Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
  # print(df.columns)


  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  df_cat = df_cat.groupby(by=['cardio', 'variable', 'value']).size().reset_index(name= 'total')

  # print('--------\n', df_cat, '--------\n')

  # Draw the catplot with 'sns.catplot()'
  g = sns.catplot(x= 'variable', y= 'total', col='cardio', data= df_cat, kind= 'bar', hue= 'value')
  fig = g.fig

  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.drop(['bmi'], axis= 1)[
      (df['ap_lo'] <= df['ap_hi']) & 
      (df['height'] >= df['height'].quantile(0.025)) & 
      (df['height'] <= df['height'].quantile(0.975)) & 
      (df['weight'] >= df['weight'].quantile(0.025)) & 
      (df['weight'] <= df['weight'].quantile(0.975))
      ]
    
    print(df_heat.describe())
    print(df_heat.info())


    # Calculate the correlation matrix
    corr = df_heat.corr()
    print(corr)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype= bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask= mask, annot=True, fmt= '0.1f')


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig

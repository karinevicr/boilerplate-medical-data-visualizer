import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('boilerplate-medical-data-visualizer/medical_examination.csv')

# 2
df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)


# 3
df['cholesterol'] = (df['cholesterol'].apply(lambda x: 0 if x == 1 else 1))
df['gluc'] = (df['gluc'].apply(lambda x: 0 if x == 1 else 1))


def draw_cat_plot():

    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["active", "alco", "cholesterol", "gluc", "overweight", "smoke"])
    

    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index =False).count()

    # sns.set_theme(style='darkgrid')
    fig = sns.catplot(x = 'variable', y = 'total', data = df_cat, kind = 'bar', 
                      col = 'cardio', hue = 'value').fig

    fig.savefig('catplot.png')
    return fig


def draw_heat_map():

    # Filtrar o DataFrame
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calcular a matriz de correlação
    corr = df_heat.corr(method='pearson')

    # Gerar uma máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Criar a figura e os eixos para o mapa de calor
    fig, ax = plt.subplots(figsize=(12, 12))

    # Criar o mapa de calor
    sns.heatmap(corr, annot=True, fmt='.1f', linewidths=1, mask=mask, square=True, center=0.8,
                cbar_kws={'shrink': 0.5})

    # Salvar a figura gerada
    fig.savefig('heatmap.png')

    # Retornar a figura
    return fig

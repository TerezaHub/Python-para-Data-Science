from google.colab import files
files.upload()
# aquisição de dados
# indicadores de desempenho. Para problemas de regressão utilizamos comumente o RMSE(raiz do erro quadrático médio). Para maiores infos acesse https://www.maxwell.vrac.puc-rio.br/3509/3509_6.PDF
# análises explortória dos dados
import pandas as pd
housing = pd.read_csv("https://gist.githubusercontent.com/qodatecnologia/36d07860f823c45ab811838349da9ff9/raw/111aa4466115c41ef34d68f53e3e31a9f73c1d4e/housing.csv")
housing.head()
# 10 features, sendo cada linha um distrito
# 20639 amostras
housing.info()
# única feature não-numérica
housing["ocean_proximity"].value_counts()
# Resumo numérico, "total_bedromms" nos mostra alguns desafios
# SKILL NECESSÁRIA: Probabilidade/Estatística
housing.describe()
# SKILL NECESSÁRIA: Probabilidade/Estatística, Dataviz, Análise de Dados
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
# preparação dos dados. Train/Test Split
!pip install scikit-learn
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head()
housing["median_income"].hist()
# "median income” está agrupada em tomo de 2 a 5 (dezenas de milhares de dólares), mas algumas rendas medianas vão muito além de 6. Isto pode enviesar nossos dados, ou seja, forçar uma tendência na hora da predição. Estratificar dados se faz necessário para que as amostras coletadas sejam proporcionais. Utilizaremos numpy para normalizar estes dados e criar novas features
# feature enginnering
import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    # dividimos os valores pelo tamanho do dataset e assim verificamos as proporções de income_cat
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# dataviz (again...)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# A localização afeta os preços destes imóveis?
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
# Suficiente para uma correlação? valores entre -1 a 1, onde 1 significa CORRELAÇÃO POSITIVA FORTE
# coeficiente de correlação Pearson
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# Correlação com PANDAS
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# Removendo nossa target dos dados de treino
housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy() # cópia na variável housing_labels
# Removendo nossa target dos dados de treino
housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy() # cópia na variável housing_labels
# Removendo nossa target dos dados de treino
housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy() # cópia na variável housing_labels
# Limpando missing values
sample_incomplete_rows.drop("total_bedrooms", axis=1)
median = housing["total_bedrooms"].median()
# Limpando missing values
sample_incomplete_rows.drop("total_bedrooms", axis=1)
median = housing["total_bedrooms"].median()
# Limpando missing values
sample_incomplete_rows.drop("total_bedrooms", axis=1)
median = housing["total_bedrooms"].median()
# Tranasformamos o TREINO com missing values substituidos pela mediana
X = imputer.transform(housing_num)
# Tranasformamos o TREINO com missing values substituidos pela mediana
X = imputer.transform(housing_num)

# Estimadores: Qualquer objeto que possa estimar alguns parâmetros com base em um conjunto de dados é chamado de estimador (por exemplo, um imputador é um estimador). À estimativa em si é realizada pelo método fit() e utiliza apenas um conjunto de dados como parâmetro (ou dois para algoritmos de aprendizado supervisionado; o segundo conjunto de dados contém os rótulos). Qualquer outro parâmetro necessário para orientar o processo de estimativa é considerado um hiperparâmetro (como a estratégia de um imputador) e deve ser definido como uma variável de instância (geralmente via parâmetro construtor).

# Transformadores: Alguns estimadores (como um imputador) também podem transformar um conjunto de dados; estes são chamados de transformadores. Mais uma vez, a API é bastante simples: a transformação é realizada pelo método transform() com o conjunto de dados para transformar como parâmetro. Retorna o conjunto de dados transformados. Essa transformação geralmente depende dos parâmetros aprendidos, como é o caso de um imputador. Todos os transformadores também têm um método de conveniência chamado fit_transform()

# PRÉ-PROCESSAMENTO DA FEATURE CATEGÓRICA "ocean proximity"
housing_cat = housing[['ocean_proximity']]
housing_cat.head()
# Lidando com dados não-numéricos https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded
# ONE HOT ENCODER
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot
# ONE HOT ENCODER
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot
# BINARIZAR
# É possível transformar inteiros em categorias, assim como categorias em números inteiros
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot
housing.columns
# TRANSFORMAÇÃO CUSTOMIZADA
# Feature Scaling é uma das mais importantes transformações de dados disponível, sendo geralmente dividida em "Min/Max scaling" e "Standardization".

# MIN/MAX SCALING Os valores são alterados e redimensionados para que acabem variando de O a 1. Fazemos isso subtraindo o valor mínimo e divídindo pelo máximo menos o mínimo. O Scikit-Leam fornece um transformador chamado MinMaxScaler para isso. Ele possui um hiperparâmetro "feature_range" que permite alterar o intervalo, se você não quiser de 0 a 1 por algum motivo.

# STANDARDIZATION Diferentemente da escala min-max, a padronização não vincula valores a um intervalo específico, o que pode ser um problema para alguns algoritmos(por exemplo, redes neurais geralmente esperam valores de 0 a 1). No entanto, a padronização é muito menos afetada pelos valores discrepantes. Por exemplo, suponha que um distrito tenha uma renda mediana igual a 100(por engano). À escala Min-max esmagaria todos os outros valores de 0 a 15 para O a 0.15, enquanto a padronização não seria muito afetada. O ScikitLeam fornece um transformador chamado StandardScaler para padronização.

# https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
# https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html

from sklearn.base import BaseEstimator, TransformerMixin

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.preprocessing import FunctionTransformer
# combinação de atributos
def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared

## O pipeline faz uma lista de pares nome/estimador, definindo uma sequência de etapas. Todos, exceto o último estimador, devem ser transformadores(ou seja, eles devem ter um método "fit_transform()"). Os nomes podem ser o que você quiser. Quando você chama o método fit() do pipeline, ele chama fit_transform() sequencialmente em todos os transformadores, passando a saída de cada chamada como parâmetro para a próxima chamada, até atingir o estimador final, para o qual chama apenas método fit().

housing_num_tr

# Modelo preditivo (treino)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

#cross - validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
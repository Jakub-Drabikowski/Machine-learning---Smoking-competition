# Smoking-competition-from-kaggle
Machine Learning - kaggle competition

Wstęp
To wydarzenie konkursowe na platformie Kaggle skupia się na tworzeniu modeli predykcyjnych, które mają identyfikować status palenia tytoniu u osób na podstawie sygnałów biologicznych. Aby poradzić sobie z tym wyzwaniem, utworzono propozycje modeli uczenia maszynowego za pomocą LazyPredict w celu wyodrębnienia najefektywniejszego modelu, który przewidywałby to, czy osoba jest paląca, czy nie. Poniżej zostaną przedstawione części kodu oraz ich opis. Implementacja algorytmów oraz pełna analiza znajduje się w osobym pliku.


## Źródło danych
[Dataset on Kaggle](https://kaggle.com/competitions/ml-olympiad-smoking) zestaw danych, który zawiera wszelkie czynniki biologiczne człowieka oraz to, czy jest to osoba paląca, czy nie.
Autorzy oraz właściciele: Rishiraj Acharya.

## Wstępna analiza danych

1. **Zainstalowanie bibliotek i wczytanie danych:**
   Ten kod importuje potrzebne biblioteki do analizy danych i uczenia maszynowego, wczytuje dane treningowe i testowe z plików CSV,
   wyświetla kilka ostatnich wierszy ramki danych treningowych, wypisuje nazwy kolumn ramki danych testowych, oraz wyświetla
   podsumowanie informacji o danych treningowych i testowych, a także sprawdza, czy istnieją brakujące wartości w danych.
     

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report pwd = os.getcwd()

train_df = pd.read_csv(pwd + '/train.csv')
train_df.tail() pwd = os.getcwd()
test_df = pd.read_csv(pwd + '/test.csv')
test_df.tail() print(train_df.columns) test_df.info() train_df.info() train_df.isnull().sum() test_df.isnull().sum() Opisz to pokrótce
```

2. **Analiza EDA
   Poniżej znajduje się kod z przykładowymi histogramami rozkładu wzrostu, wagi itp. Niżej znajduje się siatka wykresów FacetGrid pokazująca związek wieku i poziomu hemoglobiny.

```
columns_to_plot = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'Cholesterol', 'triglyceride']

plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    plt.hist(train_df[column], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Liczba obserwacji')

plt.tight_layout()
plt.show()
```
![image](https://github.com/Jakub-Drabikowski/Machine-learning---Smoking-competition/assets/83064196/824df5a5-c733-4212-8ebd-99a273b231a7)

```
g = sns.FacetGrid(train_df, col = 'smoking', hue = 'smoking', height = 5, aspect = 1.5)
g.map(sns.scatterplot, 'age', 'hemoglobin', alpha = 0.7)
```
![image](https://github.com/Jakub-Drabikowski/Machine-learning---Smoking-competition/assets/83064196/3b11e85f-d7b9-48f3-b252-d94dc557bcd9)


3. **Budowa algorytmów z pomocą LazyPredict:
  Kod ten wykorzystuje bibliotekę LazyPredict do przetestowania wielu klasyfikatorów na podstawie dostarczonych danych treningowych.

```
from lazypredict.Supervised import LazyClassifier
train_sample=train_df.sample(n=5000)
train_sample.shape
y = train_sample.pop('smoking')
X = train_sample
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)
clf = LazyClassifier(verbose = 0, predictions = True)
models, predictions = clf.fit(X_train, y_train, X_test, y_test)
models
```
![image](https://github.com/Jakub-Drabikowski/Machine-learning---Smoking-competition/assets/83064196/b683354e-3a35-4634-970f-0e02456552e8)
![image](https://github.com/Jakub-Drabikowski/Machine-learning---Smoking-competition/assets/83064196/2b4df420-c185-4570-9fb3-921b377ce0bd)


4. **Ocena działania algorytmów za pomocą classification_report
   Kod ten generuje raporty klasyfikacyjne dla predykcji różnych modeli w ramce danych predictions w odniesieniu do prawdziwych etykiet y_test.
     
```
for i in predictions.columns:
    print(i,'\n')
    print(classification_report(y_test, predictions[i]),'\n')
```

## Podsumowanie
Największą wartość, jeśli chodzi o prawidłową predykcję, czy osoba jest paląca, czy nie, osiąga algorytm LGBMClassifier.

# Smoking-competition-from-kaggle
Machine Learning - kaggle competition

Wstęp
To wydarzenie konkursowe na platformie Kaggle skupia się na tworzeniu modeli predykcyjnych, które mają identyfikować status palenia tytoniu u osób na podstawie sygnałów biologicznych. Aby poradzić sobie z tym wyzwaniem, utworzono propozycje modeli uczenia maszynowego za pomocą LazyPredict w celu wyodrębnienia najefektywniejszego modelu, który przewidywałby to, czy osoba jest paląca, czy nie.


## Źródło danych
[Dataset on Kaggle](https://kaggle.com/competitions/ml-olympiad-smoking) zestaw danych, który zawiera wszelkie czynniki biologiczne człowieka oraz to, czy jest to osoba paląca, czy nie.
Autorzy oraz właściciele: Rishiraj Acharya.

## Implementacja algorytmu wraz z EDA

1. **Zainstalowanie bibliotek i wczytanie danych:**
   - Wykorzystano biblioteki takie jak NumPy, Pandas, Matplotlib, Seaborn i TensorFlow.
   - Dane zostały pobrane za pomocą Kaggle API, a następnie przetworzone i przygotowane do modelowania.
     

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
```

2. **Przygotowanie i załadowanie danych obrazowych:
   - Dane obrazowe zostały załadowane przy użyciu ImageDataGenerator z TensorFlow.
   - Zdefiniowano generatory danych treningowych i testowych.
     

```
train_gen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
train_data = train_gen.flow_from_directory('/content/casting_data/casting_data/train',
                                          class_mode='binary',
                                          batch_size=8,
                                          target_size=(64,64),
                                          color_mode='grayscale')

test_gen = ImageDataGenerator(rescale=1/255)
test_data = test_gen.flow_from_directory('/content/casting_data/casting_data/test',
                                        class_mode='binary',
                                        batch_size=8,
                                        target_size=(64,64),
                                        color_mode='grayscale')
```

3. **Zbudowanie modelu sieci neuronowej:
   - Zdefiniowano sekwencyjny model za pomocą TensorFlow/Keras z warstwami Conv2D, MaxPooling2D, Flatten i Dense.
   - Ustalono liczbę warstw, funkcje aktywacji i rozmiary warstw.
     
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
4. **Kompilacja i trening modelu:
   - Skompilowano model z użyciem optymalizatora 'adam' i funkcji straty 'binary_crossentropy'.
   - Model został wytrenowany na danych treningowych i oceniony na danych testowych.
     
```
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(train_data, validation_data=test_data, epochs=10)
```

5. **Ocena modelu i generowanie raportu klasyfikacji:
   - Dokonano predykcji na zbiorze testowym i porównano wyniki z rzeczywistymi etykietami.
   - Przygotowano raport klasyfikacji, który zawiera precision, recall i f1-score.

```
pred_probability = model.predict_generator(test_data)
predictions = pred_probability > 0.5
print(classification_report(test_data.classes, predictions))
```

6. **Wizualizacja wyników treningu:
   - Przedstawiono wykres straty (loss) na zbiorze treningowym i testowym w kolejnych epokach.
     
![image](https://github.com/Jakub-Drabikowski/Machine-learning-kontrola-jakosci/assets/83064196/f2da8bcd-fb95-42e6-9c42-0259507e451b)

```
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
```

## Podsumowanie

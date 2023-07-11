# Çalışmanın lineer regresyonda değerlendirilmesi
# Örnek veriler ve oluşturulan veri seti kapsamında incelenebilir.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# Tabloyu DataFrame olarak yükleme
data = pd.DataFrame({
    'Puan': [100, 100, 100, 100, 100, 100, 100, 100, 98, 98, 98, 98, 97, 97, 97, 97, 96, 85, 85, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 70],
    'Polarite': [0.235, 0.332, 0.390, 0.120, 0.309, 0.222, 0.374, 0.234, 0.054, 0.250, 0.178, 0.346, 0.156, 0.267, 0.164, 0.235, 0.300, 0.089, -0.100, 0.227, 0.450, 0.203, 0.129, 0.296, -0.017, 0.120, 0.400, 0.147, 0.084, 0.035, 0.133],
    'Objektiflik': [0.537, 0.421, 0.550, 0.428, 0.682, 0.553, 0.700, 0.574, 0.452, 0.700, 0.453, 0.771, 0.326, 0.700, 0.501, 0.494, 0.488, 0.427, 0.576, 0.585, 0.700, 0.629, 0.465, 0.661, 0.388, 0.590, 0.425, 0.517, 0.574, 0.400, 0.533]
})

dataExcel = pd.read_excel('acv-user.xlsx')

X = dataExcel[['polarity', 'subjectivity']]
y = dataExcel['score']

# Regresyon modelini oluşturma ve eğitme
model = LinearRegression()
model.fit(X, y)

# Katsayıları ve kesme noktasını elde etme
coef = model.coef_
intercept = model.intercept_

# R-kare değerini elde etme
r_square = model.score(X, y)

# Model sonuçlarını yazdırma
print("Katsayılar:", coef)
print("Kesme Noktası:", intercept)
print("R-kare değeri:", r_square)

# Yeni veri için tahmin yapma
new_data = pd.DataFrame({
    'polarity': [0.2, 0.3, 0.4],
    'subjectivity': [0.5, 0.6, 0.7]
})
predictions = model.predict(new_data)
print("Tahminler:", predictions)

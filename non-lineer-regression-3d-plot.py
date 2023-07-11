# Bu alanda non-lineer regresyon analizi daha kapsamlı oluşturulmuştur.
# Tahminlenen örnek değerlere göre puanlar belirlenmiş, gerçek ve
# tahmin değerler karşılaştırılmıştır. Karşılaştırma sonucu hata payı
# ve eğitilen modelin performansı ölçülmüştür. Gösterimler 3B uzayda sunulmuştur.
# Örnek görseller kaydedilmiştir.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

data = pd.read_excel('acv-user.xlsx')

X = data[['polarity','subjectivity']]
y = data['score']

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=3)  # 2. dereceden polinomik özellikler
X_poly = poly_features.fit_transform(X)  # X veri kümesini polinomik özelliklere dönüştürme

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

# analizin 3 boyutlu gösterimi ve regresyon çizgisi
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['polarity'], X['subjectivity'], y)
ax.plot_trisurf(X['polarity'], X['subjectivity'], y_pred, color='red')
ax.set_xlabel('polarity')
ax.set_ylabel('subjectivity')
ax.set_zlabel('Puan')
plt.savefig('3d-analyse-non-lineer-regression.png', dpi=300)
plt.show()

# Gerçek puanlarla tahminlenen puanların karşılaştırılması
plt.scatter(range(len(y)), y, color='blue', label='Gerçek Puan')
plt.plot(range(len(y)), y_pred, color='red', label='Tahmin Edilen Puan')
plt.xlabel('Örnekler')
plt.ylabel('Puan')
plt.legend()
plt.savefig('3d-analyse-compare-reals-forecast.png', dpi=300)
plt.show()

# Hata payı ölçümleri
errors = y - y_pred
plt.hist(errors, bins=10)
plt.xlabel('Hata')
plt.ylabel('Frekans')
plt.savefig('3d-analyse-error-scheme.png', dpi=300)
plt.show()

# Model performans değerlendirmesi
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print("R-kare değeri:", r2)
print("MSE:", mse)
print("MAE:", mae)







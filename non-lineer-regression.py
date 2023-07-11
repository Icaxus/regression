# Doğrusal Olmayan Regresyon analizi için acv-user.xlsx dosyasındaki
# veriler kullanılmıştır. Bu verilerl metacritic.com üzerinden elde edilen
# ps5 oyunlarından derlenmiştir. Kullanılan veriler 929 satırlık olup
# siteden alınan yorumun puanlarını ve yorumdan elde edilen duygu
# analizlerini içerir (duygu kutup derecesi ve nesnellik-öznellik derecesi)
# Amaç d-o-regresyon analizi yaparken  bağımlı ve bağımsız değişkenlerini
# puan-polarity-subjectivity arasında değiştirerek verilerden çeşitli
# sonuçlar elde edileceğini görebilmek ve yapılacak yeni yorumların ise
# ne gibi değerlere yakın olacağını tahminlemektir. Bu çalışma tahminlemenin
# hangi doğrultuda gerçekleştirileceğini öngörebilmemizi sağlayacak temeli barındırmaktadır.


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#deneme verileri
data = pd.DataFrame({
    'Polarite': [0.235, 0.332, 0.390, 0.120, 0.309, 0.222, 0.374, 0.234, 0.054, 0.250, 0.178, 0.346, 0.156, 0.267, 0.164, 0.235, 0.300, 0.089, -0.100, 0.227, 0.450, 0.203, 0.129, 0.296, -0.017, 0.120, 0.400, 0.147, 0.084, 0.035, 0.133],
    'Puan': [100, 100, 100, 100, 100, 100, 100, 100, 98, 98, 98, 98, 97, 97, 97, 97, 96, 85, 85, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 70]
})

X = data[['Polarite']]
y = data['Puan']

# excel verileri burada devreye girmektedir. Deneme verileri için takip eden 34. satıra kadar
# yorum satırına alabilirsiniz.
dataExcel = pd.read_excel('acv-user.xlsx')

X = dataExcel[['subjectivity']]
y = dataExcel['polarity']

# verilerin elde edilebiliceğini görmek için alınan kısımlar.
# print(dataExcel['score'])
# exit()

# lineer olmayan bir regresyon analizi için özelliklerin tanımlanması
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

# modelin elde edilmesi ve verilere uyarlanması (modelin eğitilmesi)
model = LinearRegression()
model.fit(X_poly, y)

# Model sonuçlarını değerlendirin
y_pred = model.predict(X_poly)

# Verileri görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Veriler')
plt.xlabel('Objektiflik')
plt.ylabel('Polarite')

# Regresyon çizgisi
x_values = np.linspace(X.min(), X.max(), 100)
x_values_poly = poly_features.transform(x_values.reshape(-1, 1))
y_values = model.predict(x_values_poly)
plt.plot(x_values, y_values, color='red', label='3. Derece Regresyon')

plt.legend()
plt.savefig('third-degree-regression.png', dpi=300)
plt.show()

# Katsayıları ve kesme noktasını elde etme
coef = model.coef_
intercept = model.intercept_

# R-kare değerini elde etme
r_square = model.score(X, y)

# Model sonuçlarını yazdırma
print("Katsayılar:", coef)
print("Kesme Noktası:", intercept)
print("R-kare değeri:", r_square)

# ilgili değerlere göre polarity tahminlerini gösteren örnek verilerin durumu
new_data = pd.DataFrame({
    'subjectivity': [0.5, 0.6, 0.7]
})
new_data_poly = poly_features.transform(new_data)
predictions = model.predict(new_data_poly)
print("Tahminler:", predictions)

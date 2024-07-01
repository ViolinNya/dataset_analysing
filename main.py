import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv('london_merged.csv')

# Преобразование столбца timestamp в формат datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])


## Анализ временных рядом
# Установка индекса данных на столбец timestamp
data.set_index('timestamp', inplace=True)

# Построение графика временного ряда
plt.figure(figsize=(15, 6))
plt.plot(data['cnt'], label='Количество поездок')
plt.xlabel('Дата')
plt.ylabel('Количество поездок')
plt.title('Временной ряд количества поездок')
plt.legend()
plt.show()

# Декомпозиция временного ряда
decomposition = seasonal_decompose(data['cnt'], model='additive', period=365)

plt.figure(figsize=(15, 12))
plt.subplot(411)
plt.plot(decomposition.observed, label='Оригинальные данные')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Тренд')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Сезонность')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Остатки')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()




## Влияние погодных условий на количество поездок
# Корреляционная матрица
corr_matrix = data[['cnt', 't1', 't2', 'hum', 'wind_speed', 'weather_code']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

# Зависимость количества поездок от температуры t1
data['t1_bins'] = pd.cut(data['t1'], bins=10)
t1_grouped = data.groupby('t1_bins')['cnt'].mean()
t1_labels = [f'{interval.left:.1f}-{interval.right:.1f}' for interval in t1_grouped.index]

plt.figure(figsize=(10, 6))
t1_grouped.plot(kind='bar', color='skyblue')
plt.title('Зависимость количества поездок от температуры t1')
plt.xlabel('Температура t1 (°C)')
plt.ylabel('Среднее количество поездок')
plt.xticks(range(len(t1_labels)), t1_labels, rotation=45)
plt.show()

# Зависимость количества поездок от температуры t2
data['t2_bins'] = pd.cut(data['t2'], bins=10)
t2_grouped = data.groupby('t2_bins')['cnt'].mean()
t2_labels = [f'{interval.left:.1f}-{interval.right:.1f}' for interval in t2_grouped.index]

plt.figure(figsize=(10, 6))
t2_grouped.plot(kind='bar', color='skyblue')
plt.title('Зависимость количества поездок от температуры t2')
plt.xlabel('Температура t2 (°C)')
plt.ylabel('Среднее количество поездок')
plt.xticks(range(len(t2_labels)), t2_labels, rotation=45)
plt.show()

# Зависимость количества поездок от влажности
data['hum_bins'] = pd.cut(data['hum'], bins=10)
hum_grouped = data.groupby('hum_bins')['cnt'].mean()
hum_labels = [f'{interval.left:.1f}-{interval.right:.1f}' for interval in hum_grouped.index]

plt.figure(figsize=(10, 6))
hum_grouped.plot(kind='bar', color='skyblue')
plt.title('Зависимость количества поездок от влажности')
plt.xlabel('Влажность (%)')
plt.ylabel('Среднее количество поездок')
plt.xticks(range(len(hum_labels)), hum_labels, rotation=45)
plt.show()

# Зависимость количества поездок от скорости ветра
data['wind_speed_bins'] = pd.cut(data['wind_speed'], bins=10)
wind_speed_grouped = data.groupby('wind_speed_bins')['cnt'].mean()
wind_speed_labels = [f'{interval.left:.1f}-{interval.right:.1f}' for interval in wind_speed_grouped.index]

plt.figure(figsize=(10, 6))
wind_speed_grouped.plot(kind='bar', color='skyblue')
plt.title('Зависимость количества поездок от скорости ветра')
plt.xlabel('Скорость ветра (м/с)')
plt.ylabel('Среднее количество поездок')
plt.xticks(range(len(wind_speed_labels)), wind_speed_labels, rotation=45)
plt.show()
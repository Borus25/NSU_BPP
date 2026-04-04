import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка данных из твоих файлов
df1 = pd.read_csv('results_v1.csv')
df2 = pd.read_csv('results_v2.csv')

# Базовое время (T1) для расчета ускорения
t1_v1 = df1.loc[df1['Threads'] == 1, 'Time'].values[0]
t1_v2 = df2.loc[df2['Threads'] == 1, 'Time'].values[0]

# 2. Расчет метрик
# Ускорение S(p) = T1 / Tp
df1['Speedup'] = t1_v1 / df1['Time']
df2['Speedup'] = t1_v2 / df2['Time']

# Эффективность E(p) = S(p) / p
df1['Efficiency'] = df1['Speedup'] / df1['Threads']
df2['Efficiency'] = df2['Speedup'] / df2['Threads']

# 3. Построение графиков
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
threads = df1['Threads']

# --- График 1: Время работы ---
axes[0].plot(threads, df1['Time'], 'o-', label='Вариант 1 (Separate parallel for)')
axes[0].plot(threads, df2['Time'], 's-', label='Вариант 2 (Single parallel)')
axes[0].set_title('Время выполнения T(p)')
axes[0].set_xlabel('Число ядер')
axes[0].set_ylabel('Время, сек')
axes[0].grid(True)
axes[0].legend()

# --- График 2: Ускорение ---
axes[1].plot(threads, df1['Speedup'], 'o-', label='Вариант 1')
axes[1].plot(threads, df2['Speedup'], 's-', label='Вариант 2')
axes[1].plot(threads, threads, '--', color='gray', label='Идеальное')
axes[1].set_title('Ускорение S(p)')
axes[1].set_xlabel('Число ядер')
axes[1].set_ylabel('S = T1 / Tp')
axes[1].grid(True)
axes[1].legend()

# --- График 3: Эффективность ---
axes[2].plot(threads, df1['Efficiency'], 'o-', label='Вариант 1')
axes[2].plot(threads, df2['Efficiency'], 's-', label='Вариант 2')
axes[2].axhline(y=1.0, color='gray', linestyle='--', label='Идеальная (1.0)')
axes[2].set_title('Эффективность E(p)')
axes[2].set_xlabel('Число ядер')
axes[2].set_ylabel('E = S / p')
axes[2].set_ylim(0, 1.1)
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.savefig('omp_analysis.png')
print("Графики сохранены в файл 'omp_analysis.png'")
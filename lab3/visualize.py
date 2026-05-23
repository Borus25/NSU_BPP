import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====================================
# 1. Загрузка и подготовка данных
# ====================================
df = pd.read_csv('scaling_results.csv', sep=';')

# Извлечение числовых значений
df['MatrixSize'] = df['MatrixSize'].astype(int)
df['Cores'] = df['Cores'].astype(int)
df['Time'] = df['ExecutionTime'].str.extract(r'Time:\s*([0-9.]+)').astype(float)

# Сортировка для аккуратных графиков
df = df.sort_values(['MatrixSize', 'Cores'])

# ====================================
# 2. Вычисление ускорения и эффективности
# ====================================
# Для каждого размера матрицы находим время на 1 ядре
T1_by_size = df[df['Cores'] == 1][['MatrixSize', 'Time']].rename(columns={'Time': 'T1'})
df = df.merge(T1_by_size, on='MatrixSize')

df['Speedup'] = df['T1'] / df['Time']
df['Efficiency'] = df['Speedup'] / df['Cores']

# Идеальное ускорение (для графиков против числа процессов)
df['IdealSpeedup'] = df['Cores']

# ====================================
# 3. ГРАФИК 1: Ускорение от размера матрицы
# ====================================
plt.figure(figsize=(10, 6))
for cores in sorted(df['Cores'].unique()):
    subset = df[df['Cores'] == cores]
    plt.plot(subset['MatrixSize'], subset['Speedup'], 'o-', linewidth=2, markersize=6, label=f'{cores} процессов')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Ускорение', fontsize=12)
plt.title('Зависимость ускорения от размера матрицы', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('speedup_vs_matrix_size.png', dpi=150)
plt.show()

# ====================================
# 4. ГРАФИК 2: Время выполнения от размера матрицы
# ====================================
plt.figure(figsize=(10, 6))
for cores in sorted(df['Cores'].unique()):
    subset = df[df['Cores'] == cores]
    plt.plot(subset['MatrixSize'], subset['Time'], 'o-', linewidth=2, markersize=6, label=f'{cores} процессов')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время выполнения, сек', fontsize=12)
plt.title('Зависимость времени выполнения от размера матрицы', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('time_vs_matrix_size.png', dpi=150)
plt.show()

# ====================================
# 5. ГРАФИК 3: Эффективность от размера матрицы
# ====================================
plt.figure(figsize=(10, 6))
for cores in sorted(df['Cores'].unique()):
    subset = df[df['Cores'] == cores]
    plt.plot(subset['MatrixSize'], subset['Efficiency'], 'o-', linewidth=2, markersize=6, label=f'{cores} процессов')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Эффективность', fontsize=12)
plt.title('Зависимость эффективности от размера матрицы', fontsize=14)
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('efficiency_vs_matrix_size.png', dpi=150)
plt.show()

# ====================================
# 6. ГРАФИК 4: Ускорение от количества процессов
# ====================================
plt.figure(figsize=(10, 6))
for size in sorted(df['MatrixSize'].unique()):
    subset = df[df['MatrixSize'] == size]
    plt.plot(subset['Cores'], subset['Speedup'], 'o-', linewidth=2, markersize=6, label=f'N={size}')

# Идеальное ускорение
ideal_cores = sorted(df['Cores'].unique())
plt.plot(ideal_cores, ideal_cores, 'k--', linewidth=2, label='Идеальное ускорение')

plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Ускорение', fontsize=12)
plt.title('Зависимость ускорения от количества процессов', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('speedup_vs_cores.png', dpi=150)
plt.show()

# ====================================
# 7. ГРАФИК 5: Эффективность от количества процессов
# ====================================
plt.figure(figsize=(10, 6))
for size in sorted(df['MatrixSize'].unique()):
    subset = df[df['MatrixSize'] == size]
    plt.plot(subset['Cores'], subset['Efficiency'], 'o-', linewidth=2, markersize=6, label=f'N={size}')

plt.axhline(y=1.0, color='k', linestyle='--', label='Идеальная эффективность')
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Эффективность', fontsize=12)
plt.title('Зависимость эффективности от количества процессов', fontsize=14)
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('efficiency_vs_cores.png', dpi=150)
plt.show()

# ====================================
# 8. ГРАФИК 6: Время выполнения от количества процессов
# ====================================
plt.figure(figsize=(10, 6))
for size in sorted(df['MatrixSize'].unique()):
    subset = df[df['MatrixSize'] == size]
    plt.plot(subset['Cores'], subset['Time'], 'o-', linewidth=2, markersize=6, label=f'N={size}')

plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Время выполнения, сек', fontsize=12)
plt.title('Зависимость времени выполнения от количества процессов', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('time_vs_cores.png', dpi=150)
plt.show()

# ====================================
# 9. Вывод таблицы в консоль
# ====================================
print("\nРезультаты масштабирования (первые 10 строк):")
print(df[['MatrixSize', 'Cores', 'Time', 'Speedup', 'Efficiency']].head(10).to_string(index=False))

print("\n\nЭффективность на 16 процессах для разных N:")
eff_16 = df[df['Cores'] == 16][['MatrixSize', 'Efficiency']]
print(eff_16.to_string(index=False))
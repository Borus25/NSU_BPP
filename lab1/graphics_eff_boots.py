import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка и подготовка данных
df = pd.read_csv('results.csv', sep=';')
t_serial = df[df['Variant'] == 'Serial']['Time'].values[0]
mpi_data = df[df['Variant'].str.contains('Variant')].copy()

# Расчет метрик
mpi_data['Speedup'] = t_serial / mpi_data['Time']
mpi_data['Efficiency'] = mpi_data['Speedup'] / mpi_data['Cores']

# 2. Настройка полотна
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
colors = {'Variant1': '#1f77b4', 'Variant2': '#2ca02c'}
variants = ['Variant1', 'Variant2']

# Список параметров для отрисовки: (индекс_оси, колонка, название, формула)
metrics = [
    (0, 'Time', 'Время выполнения', '$T_p$ (сек)'),
    (1, 'Speedup', 'Ускорение', '$S_p = T_1 / T_p$'),
    (2, 'Efficiency', 'Эффективность', '$E_p = S_p / p$')
]

for ax_idx, col, title, ylabel in metrics:
    ax = axes[ax_idx]

    for var in variants:
        data = mpi_data[mpi_data['Variant'] == var]
        line, = ax.plot(data['Cores'], data[col], label=var,
                        marker='o', markersize=8, color=colors[var], linewidth=2.5)

        # Настройка смещения для подписей
        # Variant 1 -> выше (offset > 0), Variant 2 -> ниже (offset < 0)
        offset = 12 if var == 'Variant1' else -18
        va = 'bottom' if var == 'Variant1' else 'top'

        for x, y in zip(data['Cores'], data[col]):
            ax.annotate(f'{y:.2f}',
                        xy=(x, y),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center',
                        va=va,
                        fontsize=10,
                        fontweight='bold',
                        color=colors[var])

    # Специфические настройки для каждого графика
    ax.set_title(title, fontsize=15, pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Число ядер ($p$)', fontsize=12)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.grid(True, linestyle='--', alpha=0.6)

    if col == 'Speedup':
        ax.plot([1, 16], [1, 16], 'r--', alpha=0.5, label='Идеал ($S_p=p$)')
    if col == 'Efficiency':
        ax.set_ylim(0, 1.2)  # Чтобы верхние подписи не вылезали за рамку
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)

    ax.legend(loc='best')

plt.tight_layout()
plt.savefig('hpc_performance_results.png', dpi=300)
plt.show()
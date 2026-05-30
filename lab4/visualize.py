#!/usr/bin/env python3
# plot_jacobi_results.py
# Скрипт для построения графиков по результатам бенчмаркинга метода Якоби

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# ДАННЫЕ (на основе ваших результатов)
# ============================================================================
processes = [1, 2, 4, 8, 16]
time = [72.5858, 17.9206, 10.2483, 8.22164, 4.03712]
iterations = [50, 50, 50, 50, 50]
error = [8.08959e-09, 8.09289e-09, 8.09183e-09, 8.09062e-09, 8.08496e-09]

# Вычисление ускорения (относительно 1 процесса)
speedup = [time[0] / t for t in time]

# Вычисление эффективности
efficiency = [s / p for s, p in zip(speedup, processes)]

# ============================================================================
# Функция для сохранения графиков
# ============================================================================
def save_plot(fig, filename, dpi=150):
    """Сохраняет график с высоким разрешением"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")

# ============================================================================
# 1. ГРАФИК ВРЕМЕНИ ВЫПОЛНЕНИЯ
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(processes, time, 'bo-', linewidth=2, markersize=10, markerfacecolor='blue', markeredgecolor='black')
ax1.set_xlabel('Number of Processes', fontsize=12)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
ax1.set_title('Strong Scaling: Execution Time vs Number of Processes', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xticks(processes)
ax1.set_xticklabels(processes)

# Добавление значений на график
for i, (p, t) in enumerate(zip(processes, time)):
    ax1.annotate(f'{t:.2f}s', xy=(p, t), xytext=(5, 5),
                 textcoords='offset points', fontsize=9)

save_plot(fig1, 'jacobi_time.png')

# ============================================================================
# 2. ГРАФИК УСКОРЕНИЯ
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Идеальное ускорение (линейное)
ideal_speedup = processes

ax2.plot(processes, speedup, 'ro-', linewidth=2, markersize=10,
         markerfacecolor='red', markeredgecolor='black', label='Actual Speedup')
ax2.plot(processes, ideal_speedup, 'k--', linewidth=1.5, alpha=0.7, label='Ideal Speedup')

ax2.set_xlabel('Number of Processes', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('Strong Scaling: Speedup vs Number of Processes', fontsize=14)
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)
ax2.set_xticks(processes)
ax2.set_xticklabels(processes)

# Добавление значений на график
for i, (p, s) in enumerate(zip(processes, speedup)):
    ax2.annotate(f'{s:.2f}x', xy=(p, s), xytext=(5, 5),
                 textcoords='offset points', fontsize=9)

save_plot(fig2, 'jacobi_speedup.png')

# ============================================================================
# 3. ГРАФИК ЭФФЕКТИВНОСТИ
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Целевая эффективность 50%
target_efficiency = 0.5

ax3.plot(processes, efficiency, 'go-', linewidth=2, markersize=10,
         markerfacecolor='green', markeredgecolor='black', label='Actual Efficiency')
ax3.axhline(y=target_efficiency, color='orange', linestyle='--',
            linewidth=1.5, label='50% Efficiency Target')
ax3.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Ideal Efficiency (100%)')

ax3.set_xlabel('Number of Processes', fontsize=12)
ax3.set_ylabel('Efficiency', fontsize=12)
ax3.set_title('Strong Scaling: Parallel Efficiency vs Number of Processes', fontsize=14)
ax3.legend(loc='upper right', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)
ax3.set_xticks(processes)
ax3.set_xticklabels(processes)
ax3.set_ylim(0, 2.5)

# Добавление значений на график
for i, (p, e) in enumerate(zip(processes, efficiency)):
    ax3.annotate(f'{e:.2f}', xy=(p, e), xytext=(5, 5),
                 textcoords='offset points', fontsize=9)

save_plot(fig3, 'jacobi_efficiency.png')

# ============================================================================
# 4. СВОДНЫЙ ГРАФИК (все три на одной фигуре)
# ============================================================================
fig4, axes = plt.subplots(1, 3, figsize=(15, 5))

# Время
axes[0].plot(processes, time, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Processes')
axes[0].set_ylabel('Time (s)')
axes[0].set_title('Execution Time')
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log', base=2)
axes[0].set_xticks(processes)
axes[0].set_xticklabels(processes)

# Ускорение
axes[1].plot(processes, speedup, 'ro-', linewidth=2, markersize=8)
axes[1].plot(processes, ideal_speedup, 'k--', alpha=0.5, label='Ideal')
axes[1].set_xlabel('Processes')
axes[1].set_ylabel('Speedup')
axes[1].set_title('Speedup')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log', base=2)
axes[1].set_xticks(processes)
axes[1].set_xticklabels(processes)

# Эффективность
axes[2].plot(processes, efficiency, 'go-', linewidth=2, markersize=8)
axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50%')
axes[2].set_xlabel('Processes')
axes[2].set_ylabel('Efficiency')
axes[2].set_title('Parallel Efficiency')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)
axes[2].set_xscale('log', base=2)
axes[2].set_xticks(processes)
axes[2].set_xticklabels(processes)
axes[2].set_ylim(0, 2.5)

plt.tight_layout()
save_plot(fig4, 'jacobi_summary.png')

# ============================================================================
# 5. ГРАФИК ОШИБКИ (для проверки корректности)
# ============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 6))

ax5.semilogy(processes, error, 'mo-', linewidth=2, markersize=8,
             markerfacecolor='magenta', markeredgecolor='black')
ax5.set_xlabel('Number of Processes', fontsize=12)
ax5.set_ylabel('Max Error (log scale)', fontsize=12)
ax5.set_title('Solution Error vs Number of Processes', fontsize=14)
ax5.grid(True, alpha=0.3)
ax5.set_xscale('log', base=2)
ax5.set_xticks(processes)
ax5.set_xticklabels(processes)

# Порог сходимости
ax5.axhline(y=1e-8, color='red', linestyle='--', alpha=0.5, label='Convergence threshold (ε=1e-8)')
ax5.legend(fontsize=11)

save_plot(fig5, 'jacobi_error.png')

# ============================================================================
# 6. ТАБЛИЦА РЕЗУЛЬТАТОВ (текстовый вывод)
# ============================================================================
print("\n" + "=" * 70)
print("JACOBI 3D - PERFORMANCE SUMMARY")
print("=" * 70)
print(f"{'Processes':<10} {'Time(s)':<12} {'Speedup':<12} {'Efficiency':<12} {'Error':<15}")
print("-" * 70)
for i, p in enumerate(processes):
    print(f"{p:<10} {time[i]:<12.2f} {speedup[i]:<12.2f} {efficiency[i]:<12.2f} {error[i]:<15.2e}")
print("=" * 70)

# ============================================================================
# 7. ВЫВОД В CSV (для отчёта)
# ============================================================================
df = pd.DataFrame({
    'Processes': processes,
    'Time (s)': time,
    'Iterations': iterations,
    'Speedup': speedup,
    'Efficiency': efficiency,
    'Error': error
})
df.to_csv('jacobi_results.csv', index=False)
print("\nResults saved to: jacobi_results.csv")

# ============================================================================
# 8. ЗАКЛЮЧЕНИЕ
# ============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"✓ Time on 1 process: {time[0]:.2f} seconds (≥30 sec target met)")
print(f"✓ Speedup on 16 processes: {speedup[-1]:.2f}x")
print(f"✓ Efficiency on 16 processes: {efficiency[-1]:.2%}")
print(f"✓ Target 50% efficiency on 16 cores: {'✓ MET' if efficiency[-1] >= 0.5 else '✗ NOT MET'}")

if efficiency[-1] > 1.0:
    print(f"\n⚠ Note: Superlinear speedup detected (efficiency > 100%)")
    print("  This can be explained by cache effects and better data locality")

print("=" * 70)
print("\nGenerated plots:")
print("  - jacobi_time.png       (Execution time)")
print("  - jacobi_speedup.png    (Speedup with ideal line)")
print("  - jacobi_efficiency.png (Parallel efficiency)")
print("  - jacobi_summary.png    (All three plots combined)")
print("  - jacobi_error.png      (Solution error)")
print("  - jacobi_results.csv    (Raw data)")
print("\nAll files saved in current directory.")

# ============================================================================
# 9. ПОКАЗ ВСЕХ ГРАФИКОВ (если запущено в интерактивном режиме)
# ============================================================================
if __name__ == "__main__":
    plt.show()
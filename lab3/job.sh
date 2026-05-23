#!/bin/bash

# Параметры PBS
#PBS -l walltime=00:10:00
#PBS -l select=2:ncpus=8:mpiprocs=8:mem=9000m
#PBS -m be
#PBS -N matrix_2d_test

# Переход в рабочую директорию
cd $PBS_O_WORKDIR

# 1. Компиляция (убедитесь, что путь к mpicxx верен в вашей системе)
mpecxx -mpilog -o parallel_matrix src/main2.cpp

# 2. Подготовка файла результатов
RESULT_FILE="scaling_results.csv"
echo "Cores;ExecutionTime" > $RESULT_FILE

# 3. Массив количества процессов для теста
# Важно: для 2D решетки количество ядер должно позволять построить сетку (например, 4, 9, 16)
CORES=(1 2 4 8 16)

echo "Starting matrix multiplication benchmark..."
cat $PBS_NODEFILE

for p in "${CORES[@]}"
do
    echo "Running on $p cores..."
    
    # Запуск программы
    # Мы предполагаем, что программа выводит время в последней строке (как в вашем main.cpp)
    # Используем -machinefile для корректной работы на узлах кластера
    # В файле benchmark.sh измените строку получения времени:

# Запускаем программу и ищем строку, содержащую "Time:"
    OUT=$(mpirun -machinefile $PBS_NODEFILE -np $p ./parallel_matrix)

# Извлекаем только число (удаляем текст "Time:")
    TIME=$(echo "$OUT" | grep "Time:" | awk -F: '{print $2}')

# Если в коде выводится просто число (avg_elapsed), используйте фильтр по регулярному выражению (только цифры и точка):
# TIME=$(echo "$OUT" | grep -E '^[0-9.]+$')

echo "$p;$TIME" >> $RESULT_FILE
    echo "Completed $p cores: $TIME sec"
done

echo "Benchmark finished. Results saved in $RESULT_FILE"

#!/bin/bash

# Флаг для отслеживания ручного прерывания
user_interrupted=false

# Перехватываем сигнал прерывания (Ctrl+C)
trap 'user_interrupted=true; echo "Ручное прерывание - остановка"; exit 130' SIGINT

while true; do
    echo "Запускаем: python3 dp_map.py $@"
    python3 dp_map.py "$@"
    
    exit_code=$?
    
    if $user_interrupted; then
        echo "Выполнение остановлено пользователем"
        exit 130
    fi
    
    if [ $exit_code -eq 0 ]; then
        echo "Скрипт успешно завершился"
        break
    else
        echo "Аварийное завершение (код $exit_code). Перезапуск через 60 секунд..."
        sleep 60
    fi
done
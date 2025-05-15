#!/bin/bash

# Проверяем, перетащена ли папка
if [ $# -eq 0 ]; then
    echo "Перетащите папку с PNG-файлами на этот скрипт."
    exit 1
fi

# Получаем путь к перетащенной папке
input_folder="$1"

# Проверяем, что это действительно папка
if [ ! -d "$input_folder" ]; then
    echo "Указанный путь не является папкой: $input_folder"
    exit 1
fi

# Получаем имя папки для имени выходного файла
folder_name=$(basename "$input_folder")
output_file="${folder_name}.mp4"

# Проверяем наличие ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg не установлен. Установите его через Homebrew: brew install ffmpeg"
    exit 1
fi

# Создаем анимацию с помощью ffmpeg
ffmpeg -framerate 30 -pattern_type glob -i "${input_folder}/*.png" \
       -c:v libx264 -crf 12 -pix_fmt yuv420p \
       "$output_file"

echo "Анимация создана: $output_file"

import cv2
from functions import *


path = input('Введите путь к изображению, которое необходимо обработать:\n')
img_bgr = cv2.imread(path)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Перевод и bgr в rgb
x = float(input('Введите параметр x_distortion:\n'))
y = float(input('Введите параметр y_distortion:\n'))

# Запуск функции:
output_img = distortion(img, x, y)  

# Отображение результатов:
plot_result(img, output_img)

# Запуск окна со слайдерами:
print('Нужно ли открыть интерактивное окно для выбора подходящих коэффициентов?')
res = ' '
while res not in ['Yes', 'No']:
    res = input('Введите Yes или No:\n')
if res == 'No':
    print('Работа программы завершена')
else:
    print('Надо убрать бочкообразность(1) или подушкообразность(2)')
    while res not in ['1', '2']:
        res = input('Введите 1 или 2:\n')
    if res == '1':
        interctive_window(path)
    else:
        interctive_window(path, make_distortion=True)


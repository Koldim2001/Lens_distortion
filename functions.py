def get_new_coordinates(source_x, source_y, radius, x_distortion, y_distortion):
    '''
    Функция перерасчета положений пикселей:
    
    source_x: исходная координата x пикселя в исходном изображении. 
    Это значение должно быть в диапазоне от 0 до ширины исходного изображения.
    source_y: исходная координата y пикселя в исходном изображении. Э
    то значение должно быть в диапазоне от 0 до высоты исходного изображения.
    radius: радиус линзы в пикселях. 
    Это значение определяет, насколько сильно будут искажаться пиксели в зависимости от того, 
    насколько они находятся от центра линзы.
    x_distortion: параметр искажения по оси x. 
    Чем больше это значение, тем сильнее пиксели будут смещаться относительно центра изображения.
    y_distortion: параметр искажения по оси y. 
    Чем больше это значение, тем сильнее пиксели будут смещаться относительно центра изображения.

    Функция возвращает нормализованные координаты x и y пикселя в новом преобразованном изображении. 
    PS: эти координаты также могут быть за пределами границ искаженного изображения.
    '''    
    if 1 - x_distortion * (radius**2) == 0 :
        xn = source_x
    else:
        xn = source_x / (1 - (x_distortion * (radius**2)))
        
    if 1 - y_distortion * (radius**2) == 0:
        yn = source_y
    else:
        yn = source_y / (1 - (y_distortion * (radius**2)))
    return xn, yn


def distortion(img, x_distortion, y_distortion, scale_x=1, scale_y=1):
    '''
    Функция реализует процедуру изменения исходного пиксельного расположения, делая тем самым 
    операцию, соответсвующую аберрации оптической системы под названием дисторсия
    Функция на вход примимает исходное изображение в чб или rgb представлении (img). А также
    характеристики самой аберрации:
    
    x_distortion: параметр искажения по оси x. 
    Чем больше это значение, тем сильнее пиксели будут смещаться относительно центра изображения.
    y_distortion: параметр искажения по оси y. 
    Чем больше это значение, тем сильнее пиксели будут смещаться относительно центра изображения.

    Значения x_distortion, y_distortion со знаком минус говорят о том, что реализуется 
    избавление исходного изображения от бочкообразности. Если же изображение исходное 
    подушкообразное, то необходимо ставить данные значения со знаком + 
    (Fx, Fy параметры в линзе)

    scale_x и scale_y по умолчанию равны 1. Этот параметр отвечает за степень
    сжатия иходного изображения по осям (a, b параметры в линзе)
    '''
    import numpy as np
    from math import sqrt

    # Сохраним значения высоты и ширины исходного изображения
    w, h = img.shape[0], img.shape[1]
    w, h = float(w), float(h)

    # Для чб изображений сделаем копирование исходного канала 2 раза (h,w)->(h,w,3)
    if len(img.shape) == 2:
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))

    # Создадим массив нулей который будем перезаписывать в процессе выполнения функции
    result_img = np.zeros_like(img)

    # Делаем обход каждого пикселя в выходном изображении:
    for x in range(len(result_img)):
        for y in range(len(result_img[x])):

            # Нормализация корродинат x, y чтобы были в пределах [-1, 1]
            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

            #Теперь центр изображения имеет координату (0, 0), поэтому найти евклидово
            # расстояние от центра до текущего пикселя можно так:
            rd = sqrt(xnd**2 + ynd**2)

            # Определение новых координат:
            xdu, ydu = get_new_coordinates(xnd, ynd, rd, x_distortion, y_distortion)

            # Добавление рескейлинга по осям (a, b параметры в линзе)
            xdu, ydu = xdu * scale_y, ydu * scale_x

            # Возвращение значений координат из [-1, 1] в исходные значения [h, w]
            xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

            # Если пиксель находится в пределах исходного размера изображения, то проводим 
            # замену значений. В новую координату пишем (r,g,b) значение старой координаты
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                result_img[x][y] = img[xu][yu]
    
    # Переведем значения пикселей в uint8
    return result_img.astype(np.uint8)


def plot_result(img, output_img):
    '''
    Функция для построения subplot отображения было-стало
    '''
    from  matplotlib import pyplot as plt

    plt.figure(figsize=[12, 12])
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Исходное изображение')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(output_img)
    plt.title('Результат обработки')
    plt.axis('off')
    plt.show()


# Попытка создать интерактивное окно с возможность менять параметры
# y_distortion и x_distortion в реальном времени:
def interctive_window(path, make_distortion=False):
    '''
    Функция для постройки интерактивного окна с возможностью подбирать 
    коэффициенты для дисторсии с помощью слайдера
    На вход подается путь к файлу (исходное изображение)
    make_distortion = False -> коэффициенты со знаком - чтобы делать антидисторсию
    make_distortion = True -> коэффициенты со знаком + чтобы делать создать дисторсию
    '''
    import cv2
    import numpy as np

    def img_intensity_change_x(x):
        pass

    def img_intensity_change_y(y):
        pass

    # Считаем изображение:
    img_bgr = cv2.imread(path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Перевод и bgr в rgb

    # Создаем окно:
    cv2.namedWindow('LENS DISTORTION')
    val = -1 # переменная для хранения знака + или - для коэффициентов

    # Создадим sliders:
    cv2.createTrackbar('value x', 'LENS DISTORTION', 0, 20, img_intensity_change_x)
    cv2.createTrackbar('value y', 'LENS DISTORTION', 0, 20, img_intensity_change_y)

    while (1):
        # Считаем данные со слайдера:
        x = cv2.getTrackbarPos('value x', 'LENS DISTORTION')
        y = cv2.getTrackbarPos('value y', 'LENS DISTORTION')
        # Переведем в значения меньшего диапазона и со знаком 
        if make_distortion:
            val = 1  # Поменяет знак на + при входе в функцию
        joint_1 = x * 0.05 * val
        joint_2 = y * 0.05 * val
        output_img = distortion(img, joint_1, joint_2)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        cv2.putText(output_img, 'x_distortion:'+str(joint_1), (20, 20),
                    fontFace=2, fontScale=1, color=(0, 0, 220), thickness=2)
        cv2.putText(output_img, 'y_distortion:'+str(joint_2), (20, 60),
                    fontFace=2, fontScale=1, color=(0, 0, 220), thickness=2)
        cv2.imshow('LENS DISTORTION',output_img)   

        # Обновление каждый 10 мс пока не нажмется кнопка закрытие окна
        if cv2.waitKey(10) == ord('q'):
            break
    # Закрытие окна:    
    cv2.destroyAllWindows()
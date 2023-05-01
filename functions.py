









# Попытка создать интерактивное окно с возможность менять параметры y_distortion и x_distortion в реальном времени:
def interctive_window(path, make_distortion=False):
    '''
    Функция для постройки интерактивного окна с возможностью подбирать коэффициенты для дисторсии с помощью слайдера
    На вход подается путь к файлу (исходное изображение)
    make_distortion = False -> коэффициенты со знаком - чтобы делать антидисторсию
    make_distortion = True -> коэффициенты со знаком + чтобы делать создать дисторсию
    '''
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    
    def img_intensity_change_x(x):            
        pass
    def img_intensity_change_y(y):            
        pass
    
    # Считаем изображение:
    img = imageio.imread(path)
    cv2.namedWindow('LENS DISTORTION')
    val = -1 # переменная для хранения знака + или - для коэффициентов
    # Создадим sliders:
    cv2.createTrackbar('value x', 'LENS DISTORTION', 0, 20, img_intensity_change_x)
    cv2.createTrackbar('value y', 'LENS DISTORTION', 0, 20, img_intensity_change_y)

    while(1):
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
        cv2.putText(output_img, 'x_distortion:'+str(joint_1), (20, 20), fontFace=2, fontScale=1, color=(0, 0, 220), thickness=2 )
        cv2.putText(output_img, 'y_distortion:'+str(joint_2), (20, 60), fontFace=2, fontScale=1, color=(0, 0, 220), thickness=2 )
        cv2.imshow('LENS DISTORTION',output_img)    
        # Обновление каждый 10 мс пока не нажмется кнопка закрытие окна
        if cv2.waitKey(10) == ord('q'):
            break
        
    cv2.destroyAllWindows()
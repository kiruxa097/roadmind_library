import cv2
import pandas as pd
from ultralytics import YOLO

model = YOLO('yolo26n.pt')

# Вспомогательная фун-ция, чтобы контролировать нахождение объекта относительно линии
# Используем чтобы отследить пересечение линии объектом
def line_side(P, A, B):
    return (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1])*(P[0] - A[0])

# функция для быстрого создания csv файла по видео
# основные аргументы: 
# 1. adres - адрес на видео, по которому будет csv
# 2. file_name - путь, куда сохраняем полученный csv
# 3. point1 - координаты первой точки линии, через которую будем считать машины
# 4. point2 - координаты второй точки линии, через которую будем считать машины
def create_crossings(adres, file_name, point1 = (0, 0), point2 = (960, 720)):

    video = cv2.VideoCapture(adres)

    # проверка на успешное открытие файла, ловим ошибку
    if not(video.isOpened()):
        raise RuntimeError('Видео не открылось!')

    # успешность чтения нового кадра и сам кадр
    ret, image = video.read()

    # кол-во fps у video
    fps = video.get(cv2.CAP_PROP_FPS)

    # задаем начальный номер кадра, в последствии будем его += 1
    frame_id = 1

    # храним изменения центров объектов, для того чтобы засечь пересечение линии
    control_points = {}

    # список с информацией добавили ли мы этот объект, как пересек
    otchet = {}

    # массив, который в последствии станет csv
    crossing_data = []

    while ret:

        # словарь в котором по ключу - id, будем хранить информацию об объекта на текущем кадре
        prom_data = {}
        
        # !Ограничение! не передавать картинки с размерностью менее (960, 720), чтобы программа работала корректно
        image_with_new_size = cv2.resize(image, (960, 720))

        results = model.track(
            image_with_new_size,
            conf=0.2,
            persist=True,
            tracker='bytetrack.yaml'
        )

        coordinates_on_frame = results[0].boxes.xyxy

        # Находим центры объектов и сохраняем их в массив
        centers = []
        if coordinates_on_frame is not None:
            for i in range(len(coordinates_on_frame)):
                coor = coordinates_on_frame[i]

                # находим координаты центра объекта 
                xc = float((coor[0] + coor[2])/2)
                yc = float((coor[1] + coor[3])/2)

                centers.append((xc, yc))

        id_objects_on_frame = results[0].boxes.id

        # храним историю центров каждого объекта, чтобы вспоследствии засечь пересечение линии
        if id_objects_on_frame is not None:
            for i in range(len(id_objects_on_frame)):

                # базовая работа со словарями, если точка !уникальным id! уже была на кадрах, то в value есть массив, в который мы можем добавить -> тут сразу добавляем
                if control_points.get(id_objects_on_frame[i].item()):
                    control_points[id_objects_on_frame[i].item()].append(centers[i])

                # тут, так как ранее точка с !уникальным id! еще не была на кадрах, то мы сначала для нее создаем массив, в который сразу после добавляем центр объекта
                else:
                    control_points[id_objects_on_frame[i].item()] = []
                    control_points[id_objects_on_frame[i].item()].append(centers[i])

        # прохожусь по id объектов, так как это позволит не делать лишних итераций по массиву, когда найдем объект, который пересек контрольную линию
        if id_objects_on_frame is not None:
            for i in range(len(id_objects_on_frame)):
                prom_data[int(id_objects_on_frame[i].item())] = {
                    'frame_id' : frame_id,
                    'timestamp_sec' : (frame_id-1)/fps,
                    'track_id' : int(id_objects_on_frame[i].item()),
                    'class_name' : results[0].names[int(results[0].boxes.cls[i].item())],
                    'confidence' : float(results[0].boxes.conf[i]),
                    'x1' : float(results[0].boxes.xyxy[i][0]),
                    'y1' : float(results[0].boxes.xyxy[i][1]),
                    'x2' : float(results[0].boxes.xyxy[i][2]),
                    'y2' : float(results[0].boxes.xyxy[i][3]),
                    'center_x' : (float(results[0].boxes.xyxy[i][0]) + float(results[0].boxes.xyxy[i][2]))/2,
                    'center_y' : (float(results[0].boxes.xyxy[i][1]) + float(results[0].boxes.xyxy[i][3]))/2
                }

        # итерируем по ключ:значение в контрольных точках, чтобы проверить не появились ли новые объекты, которые пересекли линию
        for i in control_points.items():
            if len(i[1]) > 1:
                x1, y1 = i[1][-2]
                x2, y2 = i[1][-1]
                s1 = line_side((x1, y1), point1, point2)
                s2 = line_side((x2, y2), point1, point2)
                if (s1 > 0 and s2 <= 0) or (s1 < 0 and s2 >= 0):
                    if i[0] not in otchet and i[0] in prom_data:
                        crossing_data.append(prom_data[i[0]])
                        otchet[i[0]] = True

            # не храним информацию о более чем 2 старых точек
            if len(i[1]) > 2:
                i[1].pop(0)

        # успешность чтения нового кадра и сам новый кадр
        ret, image = video.read()

        # т.к переходим на след. кадр, номер увеличаем
        frame_id += 1

    df = pd.DataFrame(crossing_data)
    df.to_csv(file_name, index=False)

    video.release()

    return df
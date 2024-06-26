
# Загрузка данных:

Загружаем набор данных Iris, содержащий 4 признака для каждого из 150 ирисов, и их метки классов.

## PCA:
Применяем PCA для уменьшения размерности данных до 2 компонент. Это позволяет нам найти две новые оси (главные компоненты), которые объясняют наибольшую дисперсию в данных.

## ICA:

- Применяем ICA для уменьшения размерности данных до 2 компонент. ICA ищет компоненты, которые статистически независимы друг от друга.

## Создание графиков:

- Исходные данные: Двумерный график, использующий первые два признака исходных данных.
- Данные после применения PCA: Двумерный график, использующий две главные компоненты, найденные PCA.

### Настройка графиков:
- Для каждого графика добавляем подписи осей и легенды, чтобы показать, какие классы ирисов представлены в данных.
### Что мы видим на графиках:
- Исходные данные: Мы видим распределение ирисов по первым двум признакам. График может показывать некоторые перекрытия между классами, что может затруднить их различие.

- Данные после PCA: На этом графике данные преобразованы таким образом, что новые оси (главные компоненты) максимизируют дисперсию данных. Это может сделать различия между классами более очевидными и облегчить их классификацию.
- Данные после ICA: На этом графике данные преобразованы таким образом, что новые оси (независимые компоненты) максимально разнесены по мере независимости. Это может помочь лучше различать классы, если в данных присутствуют независимые источники сигналов.

### Этот пример демонстрирует, как:
- PCA помогает уменьшить размерность данных, сохраняя при этом важную информацию о структуре данных.
-  ICA преобразует данные, выделяя независимые компоненты, что может быть полезно для задач, связанных с выделением скрытых факторов или источников в данных.


## Немного теории:

- Под "компонентами" в контексте методов уменьшения размерности, таких как PCA и ICA, подразумеваются новые переменные, полученные в результате линейного преобразования исходных данных. Эти новые переменные или компоненты позволяют представить данные в новом пространстве с меньшим числом измерений, сохраняя при этом как можно больше информации о структуре данных.
В случае PCA (Principal Component Analysis):
Главные компоненты — это линейные комбинации исходных признаков, которые максимизируют дисперсию данных. Первые несколько главных компонент захватывают наибольшую часть разброса данных, позволяя эффективно уменьшить размерность при минимальных потерях информации.
В случае ICA (Independent Component Analysis):
Независимые компоненты — это линейные комбинации исходных признаков, которые максимизируют статистическую независимость между компонентами. ICA используется для разделения смешанных сигналов на их независимые источники.
Компоненты в данном контексте — это новые оси или направления в преобразованном пространстве данных, которые позволяют упростить анализ и визуализацию, а также улучшить производительность моделей машинного обучения, работая с меньшим числом признаков.


## Что же такое PCA:
- PCA — это «анализ главных компонент; PCA) — это метод многомерного анализа, который синтезирует небольшое количество некоррелированных переменных, называемых главными компонентами, которые наилучшим образом представляют общую вариацию из большого числа коррелированных переменных[1]. Используется для уменьшения размерности данных. Преобразование, дающее главный компонент, выбирается таким образом, чтобы максимизировать дисперсию при условии, что первый главный компонент максимален, а последующий главный компонент ортогонален ранее определенному главному компоненту.


- Если говорить абстрактно, то это метод уменьшения размерности, который извлекает основные компоненты, чтобы максимально не потерять исходную информацию. Входные параметры → Большое количество коррелированных переменных
Выходные данные → Небольшое количество некоррелированных переменных
→ Наилучшее представление исходной информации (математически дисперсия исходных данных)

Корреляция — это наличие линейной зависимости, например, x1=3*x2.

## КАК:
Трехмерный → двумерный

1. Определите ось первого главного компонента, которая может максимизировать дисперсию. (Максимизация дисперсии здесь означает, что расстояние между каждой точкой максимизируется, когда точка проецируется на ось.) Причина, по которой дисперсия максимальна, заключается в том, что исходная информация может быть воспроизведена. ）
2. Найдя ось первого главного компонента, определите ось второго главного компонента, которая может максимизировать дисперсию (перпендикулярную первому) во втором. Причина, по которой я сразу обращаюсь к нему, заключается в том, что он в наибольшей степени сохраняет и представляет исходную информацию.

3. Преобразуйте трехмерные точки в двумерные точки с помощью математических формул.
Например, (x1, x2, x3) — координаты точки в трёх измерениях, а (y1, y2) — точка в двух измерениях после преобразования.
4. Коэффициент вклада — это процент, который относится к тому, насколько хорошо первый главный компонент представляет исходные данные.
Можно сказать, что первый главный компонент охватывает разброс данных во всех данных.
5. Кумулятивный вклад
В основном он используется для проверки того, какой процент данных представлен компонентами. Например, если вы хотите создать компонент, который воспроизводит более 95 % данных, вы можете сказать: «Совокупный коэффициент вклада составляет 80 % до третьего главного компонента, а четвертый главный компонент — 95 %, поэтому давайте выберем до четвертого главного компонента».
6. Оценка главного компонента — это числовое значение, полученное путем применения фактических данных к первому главному компоненту. Например, если x1, x2 и x3 данных применить из 3D и получить большое число независимо от + или -, то можно увидеть, что связь с главным компонентом i высокая.

### ПОЧЕМУ:
1. Визуализация данных проста за счет уменьшения размерности. Пятимерные данные трудно увидеть, но двумерные данные (x, y) легко понять.
2. Потому что вычислительные затраты слишком высоки. 300 переменных (x1, x2, x3... x300) и 30 (x1, x2... x30) переменных, и обе могут выражать практически один и тот же результат (y), то лучше выбрать 30-мерную, то есть с 30 переменными.
3. Чтобы избежать проклятия размеренности. (Если бы я перевел это буквально, это выглядело бы как кухня...) Проклятие измерений относится к потребности в данных по мере увеличения каждого измерения. Например, предположим, что у вас есть 10 признаков и 100 обучающих данных. Поэтому мы увеличили количество функций до 100. В настоящее время легко представить, что произойдет переобучение, если исходные обучающие данные останутся на уровне 100. Это проклятие измерений. Чем больше у вас измерений, тем больше данных вам нужно. Чтобы этого избежать, требуется уменьшение размерности.

### ГДЕ:
В принципе, PCA имеет множество применений. Если вы не хотите убирать какой-то особенно независимый элемент, такой как ICA, о котором будет рассказано позже, у меня сложилось впечатление, что PCA используется почти всегда. Например, на фондовом рынке, если есть десятки вещей, на которые следует обращать внимание при инвестировании, будет легче управлять риском, извлекая только важные элементы с помощью PCA и сжимая их до нескольких. Он также используется в неврологии.





https://qiita.com/Hatomugi/items/d6c8bb1a049d3a84feaa



Уменьшение размерности данных с помощью методов, таких как PCA (Principal Component Analysis) и ICA (Independent Component Analysis), подразумевает преобразование исходных данных с высокой размерностью (многими признаками) в новое пространство с меньшим числом измерений (меньше признаков), сохраняя как можно больше информации о структуре данных. Это позволяет упростить анализ данных, снизить вычислительные затраты и улучшить работу алгоритмов машинного обучения.

Принципы уменьшения размерности
PCA (Principal Component Analysis)
PCA ищет линейные комбинации исходных признаков, которые максимизируют дисперсию данных. Эти линейные комбинации называются главными компонентами.

Первый главный компонент: направление в пространстве исходных данных, вдоль которого дисперсия данных максимальна.
Второй главный компонент: направление, перпендикулярное первому компоненту, вдоль которого дисперсия данных максимальна, и так далее.
Каждая последующая компонента объясняет оставшуюся дисперсию данных, максимально уменьшая корреляцию между компонентами. Обычно выбирают первые несколько компонент, которые объясняют наибольшую часть дисперсии данных.

ICA (Independent Component Analysis)
ICA ищет линейные комбинации исходных признаков, которые максимизируют статистическую независимость между компонентами.

Независимые компоненты: направления, в которых данные являются максимально независимыми друг от друга. Эти компоненты особенно полезны для задач, связанных с разделением источников сигналов, таких как разделение звуковых сигналов, изображений и т. д.
#   P C A a n d I C A  
 
|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Моделирование динамики физических систем с помощью Physics-Informed Neural Networks
    :Тип научной работы: M1P
    :Автор: Александр Иванович Богданов
    :Научный руководитель: степень, Фамилия Имя Отчество
    :Научный консультант(при наличии): степень, Фамилия Имя Отчество

Abstract
========

В работе решается задача улучшения качества моделирования динамики сложных механических систем с помощью нейронных сетей. Большинство подходов, оперирующих лишь набором обобщённых координат в качестве входа, не способны предсказывать поведение системы с высокой точностью. В работе исследуется Нётеровская Лагранжева нейронная сеть, которая учитывает закон сохранения энергии, импульса, момента импульса и способна восстанавливать лагранжиан системы по траекториям её движения. Вместо полносвязанных слоев в сети используются модификации внутренней архитектуры, основанные на свёрточных и рекуррентных нейронных сетях. Сравнение результатов моделирования проводилось на искусственно сгенерированных данных для пружинной системы. Результаты подтверждают то, что качество модели увеличивается, если она обладает информацией о системе.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.

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

    :Название исследуемой задачи: Моделирование динамики физических систем с помощью лагранжевых нейронных сетей
    :Тип научной работы: M1P
    :Автор: Александр Иванович Богданов
    :Научный руководитель: Вадим Викторович Стрижов
    :Научный консультант: Святослав Константинович Панченко

Abstract
========

В работе решается задача классификации траекторий физических систем. Квазипериодическая динамика системы аппроксимируется лагранжианом, который восстанавливается по обобщённым координатам с помощью Лагранжевой нейронной сети. Показано, что для параметров таких сетей выполняется гипотеза компактности: векторы параметров, соответствующие траекториям различных классов, оказываются разделимы в своём пространстве. Проводится эксперимент на датасете PAMAP2, результаты которого подтверждают, что параметры лагранжевых нейронных сетей действительно являются информативным признаковым описанием для задачи классификации.
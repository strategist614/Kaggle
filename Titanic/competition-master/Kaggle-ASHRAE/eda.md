# 数据分析

- Site Usage

site 0 meter 0在６月前存在异常０值， site 6 meter 1 在９月前和11月后存在异常０值，site 15 在2月3月存在数据缺失。



- 时间调整

https://www.kaggle.com/patrick0302/locate-cities-according-weather-temperature

 https://www.kaggle.com/frednavruzov/aligning-temperature-timestamp 

```python
locate = {
    0: {'country': 'US', 'offset': -4},
    1: {'country': 'UK', 'offset': 0},
    2: {'country': 'US', 'offset': -7},
    3: {'country': 'US', 'offset': -4},
    4: {'country': 'US', 'offset': -7},
    5: {'country': 'UK', 'offset': 0},
    6: {'country': 'US', 'offset': -4},
    7: {'country': 'CAN', 'offset': -4},
    8: {'country': 'US', 'offset': -4},
    9: {'country': 'US', 'offset': -5},
    10: {'country': 'US', 'offset': -7},
    11: {'country': 'CAN', 'offset': -4},
    12: {'country': 'IRL', 'offset': 0},
    13: {'country': 'US', 'offset': -5},
    14: {'country': 'US', 'offset': -4},
    15: {'country': 'US', 'offset': -4},
}
```

- 缺失&零值数据

可视化： https://www.kaggle.com/ganfear/missing-data-and-zeros-visualized 

可视化： https://www.kaggle.com/juanmah/ashrae-zeros 

补全方法： https://www.kaggle.com/juejuewang/handle-missing-values-in-time-series-for-beginners 

0值讨论： https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054#latest-663612 

天气数据补全：

- 数据泄露

 https://www.kaggle.com/c/ashrae-energy-prediction/discussion/116773#latest-671028 

 https://www.kaggle.com/gunesevitan/ashrae-ucf-spider-and-eda-full-test-labels 

 https://www.kaggle.com/mpware/ucf-data-limited-leakage 

- 节假日

 https://www.kaggle.com/c/ashrae-energy-prediction/discussion/115256#663029 

 https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113286 

- 外部数据

 https://www.kaggle.com/c/ashrae-energy-prediction/discussion/112841#latest-671680 

 https://www.kaggle.com/yamsam/new-ucf-starter-kernel 

- 2016年新修建建筑物

 https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113254#latest-667876 

- 最佳单模型

- 有见解的讨论

rank13：  https://www.kaggle.com/c/ashrae-energy-prediction/discussion/116773#latest-671402 

- 深度模型

  fastai：https://www.kaggle.com/poltigo/ashrae-fastai

  keras： https://www.kaggle.com/isaienkov/keras-nn-with-embeddings-for-cat-features-1-15 
  
  

- nice EDA

   https://www.kaggle.com/vikassingh1996/ashrae-great-energy-insightful-simple-eda 

   https://www.kaggle.com/juanmah/ashrae-degree-hours 



- 速度

   https://www.kaggle.com/corochann/ashrae-feather-format-for-fast-loading 

- cv insight

   https://www.kaggle.com/kyakovlev/ashrae-cv-options 

- 缺失处理

   https://www.kaggle.com/hmendonca/clean-weather-data-eda 

   https://www.kaggle.com/hmendonca/clean-weather-data-eda 
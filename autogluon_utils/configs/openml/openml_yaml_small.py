
defaults_yaml_small = ('small', '''---

- name: __defaults__
  folds: 10
  max_runtime_seconds: 3600
  cores: 8
  min_vol_size_mb: 16384
''')


dataset_yaml_small_dict = {
    'Australian':
'''- name: Australian
  openml_task_id: 146818
  metric:
    - auc
    - acc
''',
    'blood-transfusion':
'''- name: blood-transfusion
  openml_task_id: 10101
  metric:
    - auc
    - acc
''',
    'car':
'''- name: car
  openml_task_id: 146821
  metric:
    - logloss
    - acc
''',
    'christine':
'''- name: christine
  openml_task_id: 168908
  metric:
    - auc
    - acc
''',
    'cnae-9':
'''- name: cnae-9
  openml_task_id: 9981
  metric:
    - logloss
    - acc
''',
    'credit-g':
'''- name: credit-g
  openml_task_id: 31
  metric:
    - auc
    - acc
''',
    'dilbert':
'''- name: dilbert
  openml_task_id: 168909
  metric:
    - logloss
    - acc
''',
    'fabert':
'''- name: fabert
  openml_task_id: 168910
  metric:
    - logloss
    - acc
''',
    'jasmine':
'''- name: jasmine
  openml_task_id: 168911
  metric:
    - auc
    - acc
''',
    'kc1':
'''- name: kc1
  openml_task_id: 3917
  metric:
    - auc
    - acc
''',
    'kr-vs-kp':
'''- name: kr-vs-kp
  openml_task_id: 3
  metric:
    - auc
    - acc
''',
    'mfeat-factors':
'''- name: mfeat-factors
  openml_task_id: 12
  metric:
    - logloss
    - acc
''',
    'phoneme':
'''- name: phoneme
  openml_task_id: 9952
  metric:
    - auc
    - acc
''',
    'segment':
'''- name: segment
  openml_task_id: 146822
  metric:
    - logloss
    - acc
''',
    'sylvine':
'''- name: sylvine
  openml_task_id: 168912
  metric:
    - auc
    - acc
''',
    'vehicle':
'''- name: vehicle
  openml_task_id: 53
  metric:
    - logloss
    - acc
'''
}

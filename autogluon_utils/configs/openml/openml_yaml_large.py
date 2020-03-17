
defaults_yaml_large = '''---

- name: __defaults__
  folds: 10
  max_runtime_seconds: 14400
  cores: 8
  min_vol_size_mb: 16384
'''


dataset_yaml_large_dict = {
    'Airlines':
'''- name: Airlines
  openml_task_id: 189354
  metric:
    - auc
    - acc
''',
    'Albert':
'''- name: Albert
  openml_task_id: 189356
  metric:
    - auc
    - acc
''',
    'Covertype':
'''- name: Covertype
  openml_task_id: 7593
  metric:
    - logloss
    - acc
''',
    'Dionis':
'''- name: Dionis
  openml_task_id: 189355
  metric:
    - logloss
    - acc
'''
}

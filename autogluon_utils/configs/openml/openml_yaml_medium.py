
defaults_yaml_medium = ('medium', '''---

- name: __defaults__
  folds: 10
  max_runtime_seconds: 14400
  cores: 8
  min_vol_size_mb: 16384
''')


dataset_yaml_medium_dict = {
    'adult':
'''- name: adult
  openml_task_id: 7592
  metric:
    - auc
    - acc
''',
    'Amazon_employee_access':
'''- name: Amazon_employee_access
  openml_task_id: 34539
  metric:
    - auc
    - acc
''',
    'APSFailure':
'''
- name: APSFailure
  openml_task_id: 168868
  metric:
    - auc
    - acc
''',
    'bank-marketing':
'''
- name: bank-marketing
  openml_task_id: 14965
  metric:
    - auc
    - acc
''',
    'connect-4':
'''
- name: connect-4
  openml_task_id: 146195
  metric:
    - logloss
    - acc
''',
    'Fashion-MNIST':
'''- name: Fashion-MNIST
  openml_task_id: 146825
  metric:
    - logloss
    - acc
''',
    'guiellermo':
'''
- name: guiellermo
  openml_task_id: 168337
  metric:
    - auc
    - acc
''',
    'Helena':
'''
- name: Helena
  openml_task_id: 168329
  metric:
    - logloss
    - acc
''',
    'higgs':
'''
- name: higgs
  openml_task_id: 146606
  metric:
    - auc
    - acc
''',
    'Jannis':
'''
- name: Jannis
  openml_task_id: 168330
  metric:
    - logloss
    - acc
''',
    'jungle_chess_2pcs_raw_endgame_complete':
'''
- name: jungle_chess_2pcs_raw_endgame_complete
  openml_task_id: 167119
  metric:
    - logloss
    - acc
''',
    'KDDCup09_appetency':
'''
- name: KDDCup09_appetency
  openml_task_id: 3945
  metric:
    - auc
    - acc
''',
    'MiniBooNE':
'''
- name: MiniBooNE
  openml_task_id: 168335
  metric:
    - auc
    - acc
''',
    'nomao':
'''
- name: nomao
  openml_task_id: 9977
  metric:
    - auc
    - acc
''',
    'numerai28.6':
'''
- name: numerai28.6
  openml_task_id: 167120
  metric:
    - auc
    - acc
''',
    'riccardo':
'''
- name: riccardo
  openml_task_id: 168338
  metric:
    - auc
    - acc
''',
    'Robert':
'''
- name: Robert
  openml_task_id: 168332
  metric:
    - logloss
    - acc
''',
    'Shuttle':
'''
- name: Shuttle
  openml_task_id: 146212
  metric:
    - logloss
    - acc
''',
    'Volkert':
'''
- name: Volkert
  openml_task_id: 168331
  metric:
    - logloss
    - acc
''',
}

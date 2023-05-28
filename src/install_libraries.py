import pip
pip.main(['install', 'numpy'])
pip.main(['install', 'adjustText'])
pip.main(['install', 'nbconvert'])
pip.main(['install', 'pandas'])
pip.main(['install', 'seaborn'])
pip.main(['install', 'scipy'])
pip.main(['install', 'scikit-learn'])
pip.main(['install', 'statsmodels'])
pip.main(['install', 'xgboost'])
pip.main(['install', 'openpyxl'])

import subprocess
subprocess.call(['pip3', 'install', 'torch'])
subprocess.call(['pip3', 'install', 'torchvision'])
subprocess.call(['pip3', 'install', 'torchaudio'])
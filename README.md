# Setup
```
pip install -r requirements.txt
pip3 install torch torchvision torchaudio
```
or alternatively, directly run https://github.com/nbakas/nbml/blob/main/src/install_libraries.py in Python

# How to use
1) Put all *.py files in any "ROOT_DIR" folder. Under "ROOT_DIR" there should be only one working xlsx file with the above Figure's structure.
2) Start from \_\_nbml\_\_.py and run block by block.
3) If you want to predict for new out-of-sample data, put the corresponding excel file in the "Predict" folder under ROOT_DIR, and run the <Open Dataset> part of the code first

# Users' manual 
https://github.com/nbakas/nbml/blob/main/docs/__nbml__.pdf

for installation see Section 2.2

# To run this on a Cluster
https://github.com/nbakas/nbml/blob/main/hpc/hpc_notes.md

# To run this on google Colab
1. Upload the code on https://colab.research.google.com/ 2. Split the code into cells (optional) 3. Upload all *.py files 4. Upload your dataset 5. Run the code!

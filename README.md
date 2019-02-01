# dateConverter
date converter that utilizes deep learning

Takes in a list of date strings in any format, outputs a list of date strings in ANSI format. 

simply put from date_converter_2 import DateConvert

then DateConvert(["list", "of", "strings of dates in any format"])

the function should return a list of strings in ANSI format YYYY-MM-DD

Right now the script must run all the way through model compiling, training, then prediction.
Working on how to save then load LSTM with attention model that loads state of trainied model. Then it will work much faster.

REQUIRED PACKAGES
python 2
keras
numpy
argparse
faker
random
tqdm
babel
matplotlib

you can pip install all of the above. 

Derived from  oursera LSTM model with attention. Working on trying to 

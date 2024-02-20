this folder consists of:
1. Python file:
    1. <data_processing_tools.py>
    2. <main_preprocess.py>



* functions in "data_processing_tools.py" "script:
    a func. to call other function in the needed order                                                                                         <split_then_img2block>
    a func. to divide input image into 224*224 blocks                                                                                          <img2blocks>
    a func. to process input image exclude other frames if exist in case ".gif" formates                                                       <processImage>







* Note:
    Inut images path must be provided in main_preprocess file line 5 "src_path" default is"original_dataset"
    output path must be spicified in main_processing file line 6 "output_path" default is"dataset/All/"



Usage:
1. conda activate pytorch_p39
2. python /location of the file/main_preprocess.py


 










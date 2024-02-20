this folder consists of:
1. Python file:
    1. <CNN.py>
    2. <main.py>



* functions in "CNN.py" "script:
    a func. to create output text file with training details                                                                                   <create_out_txt_file>
    a func. to loa images from a given path and split into train/valid/test then convert to dataloader of tensors                              <load_data>
    a func. to write correlated RF events of a specific image to the database                                                                  <write_aquired_RF_event_records_v2>
    a func. to excute any given query                                                            <excute_query>
    a func. to correlate Image degredation errors and RF events                                  <correlate_ml_rf>
    a func. to fetch all correlated RF events for a specific image                               <get_correlated_rf_events>
    a func. to localize image degredation errors on original image                               <run_localize_on_original>
    a func. to convert a given pixel into date and time                                          <from_pixl_to_time_frame>
    a func. to localize image degredation errors on partitioned images                           <run_localization>
    a func. to get the bounding box for white and black type image degredation                   <get_white_black_bounding_boxs>
    a func. to get localtion of point type image degredation error                               <get_scatter_location>
    a func. to call localize on original and localize partitions                                 <localize>
    a func. to invoke sagemaker deployed endpoint to classify a given partitioned image          <classify_fn>
    a func. to convert images from other formats (ex: gif) to png                                <covert_image_format_png>
    a func. to partition the whole image into sub images                                         <partition_image_to_sub_images>
    a func. to call other partitioning function when needed                                      <main_partition_fn>
    a func. to copy images between s3 prifexs                                                    <copy_from_s3_to_s3>
    a func. to list content of a given s3 prifexs                                                <list_s3_files>
    a func. to upload all content of a given local path to a given s3 prefix                     <uploading_to_s3>
    a func. to download images from s3 to a given local path                                     <load_image_and_save_local>
    a func. to upload a given file to a given s3 prefix                                          <save_to_s3>
    a func. to do processing on the original image like adding frame and text                    <main_processing_main_img>
    a func. to do processing on the partitioned image like adding frame and text                 <main_processing_sub_img>






* Note:
    Input images must be inside a folder with the name "../input_data" and divided into 2 subfolder "error" and "no_error"
    All images should be 224*224 

Output:
1. Model will be saved in folder "../models/"modeltype", example "../models/ResNet34"
2. textfile will be saved in the same folder with the name "Model type_optimizer used_learning rate" excample "ResNet34_Adam_0001" file includes training info:
    1. traning loss and validation loss for each epoch
    2. Confusion Matrix 



Usage:
1. conda activate pytorch_p39
2. python /PATH/OF/main.py
example :
/home/ubuntu/ML_project/stage2/training_scripts/main.py
/home/ubuntu/ML_project/stage2/models/
/home/ubuntu/ML_project/stage2/input_data/


* Environment Setup:
    1. Conda version is "conda 4.8.3"
    2. AWS instance type is "p2.xlarge"




* Model Info:
    1. We used transfer learning technique with "ResNet34" and "ResNet50" Models, freezed all conv layers and created new fully connected layer to be trained.
    2. hyperparamers were as follows:
        1. batch_size = 20
        2. lr = 0.0001
        3. num_epochs = 40
        4. "Adam" Optimizer 

 










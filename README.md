# JDD_CCA

The article is being submitted to "The Visual Computer". The title is: Frequency-Decoupled and Bionic-Inspired RAW Image Enhancement for Low-Light Conditions.

The datasets used in this article are [Sony](https://drive.google.com/file/d/1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx/view) and [MCR](https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view), and the code will be open source after acceptance.

# Train
To train the Sony model, run train_WGCAN_Sony_L1_WaveLoss.py. For example: 
 ```
python train_WGCAN_Sony_L1_WaveLoss.py --data_dir /data/Zhaobo/SID/Sony --gt_dir /data/Zhaobo/SID/Sony/Sony/long/ --result_dir /data/Zhaobo/result/Sony/train_eval/L1_WaveLoss_sum_weights
 ```
The same principle applies to MCR.
# Test
To test the Sony model, run test_sony_metrics.py. For example: 
 ```
python test_sony_metrics.py --model bestssimmodel.pth --result_dir /data/Zhaobo/result/Sony/train_eval/L1_WaveLoss_sum_weights
 ```
In the code, the path to the dataset is specified on line 33. The same principle applies to MCR.

# Pretrained model

The pre trained model will be open sourced after the paper is received.

# Grateful

 Part of the code in this warehouse is written with reference to [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark) and [RetinexRawMamba](https://github.com/Cynicarlos/RetinexRawMamba). We are very grateful for their open source help!

 

# DoyleyResearch  
## Resource:  
Keras Documentation:  
https://keras.io/  
how to load data using from\_derictory:  
http://note4lin.top/post/keras_dataload/  
how to customize own dataloader:  
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly  

## Logs:  
----------------------------------------------------------------------------  
Data Spilt Manually, ignore similarity in one folder.  

current 2020.02.26:  
model: Unet_Pretrained_bce_jaccard_loss_iou_score_tuned.hdf5  
loss: 0.2376 - iou_score: 0.7822 - val_loss: 0.2047 - val_iou_score: 0.7883
result: See test folder.

current 2020.02.26:  
model: Unet_Pretrained_bce_jaccard_loss_iou_score_tuned_10epoches.hdf5  
loss: 0.1709 - iou_score: 0.8435 - val_loss: 0.1126 - val_iou_score: 0.8510  
result: See test folder.  
problem: how to split dataset?(may overfit now).  

---------------------------------------------------------------------------  
2020.03.12:  
Start 5 fold splits, based on folder(80% folders for train, 20% folders val)  
Problem: image ratio for train and val may not be 4:1, due to the number of images in each folder various.  

2020.03.17:  
Fold 1, 5 epoches:  
loss: 0.1300 - iou_score: 0.8805 - val_loss: 0.8275 - val_iou_score: 0.2782  
Problem: Not Robust. Overfit training data, but fails on val data.  
Solution: 1. Add augmentations. 2. Data cleaning(Remove similar data). 3. Data preprocessing.  





# CT-COVID19-Classification

model_weights:https://drive.google.com/file/d/1Qir1X1VGBybYIrjN3NRwMh0vv839-L3h/view?usp=sharing


1. open preprocess.ipynb and chage test_folder_path to your test data folder path

2.run preprocess.ipynb

3.run get_slice_range.ipynb to get slice range of every ct case

4.run effb3a-inference.ipynb get effb3a model prediction

5.run get_embedding.ipynb get every slice embedding for lstm model and swin model

6.run lstm model.ipynb get lstm model prediction

7.get swin transformer prediction from swin_inference

8.run get_ECCV_pred_from_prob.ipynb get ECCV submission 

9.ECCV submission is on covid_pred

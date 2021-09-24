# Flow-based-Pytorch

## NICE
paper : https://arxiv.org/pdf/1410.8516.pdf  
Model : https://github.com/yhy258/Flow-based-Pytorch/blob/master/NICE.py  

설명 :   


후기 :  
![gaussian_NICE](https://github.com/yhy258/Flow-based-Pytorch/blob/master/images/NICE2_gaussian.png?raw=true)
![logistic_NICE](https://github.com/yhy258/Flow-based-Pytorch/blob/master/images/NICE2_logistic.png?raw=true)
  
EPOCH : 50 train 후 sampling 결과. 왼쪽 이미지는 Gaussian distribution을 prior로 둔 경우이고, 오른쪽은 Logistic Distribution을 prior로 둔 경우이다.  
논문에 명시 된 대로 logistic distribution을 prior로 뒀을 때 더 좋은 결과를 내놓았다.  
  
flow based GM 중 시작점이 되는 NICE라는 모델은 굉장히 단순한 메커니즘을 기반으로 했다. 특히 coupling layer를 알맞게 구성하여 flow based method가 성립되도록 했을 때 좀 신기했다.

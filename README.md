# Flow-based-Pytorch

## NICE
paper : https://arxiv.org/pdf/1410.8516.pdf  
Model : https://github.com/yhy258/Flow-based-Pytorch/blob/master/NICE.py  

설명 : https://deepseow.tistory.com/45  


후기 :  
![gaussian_NICE](https://github.com/yhy258/Flow-based-Pytorch/blob/master/images/NICE2_gaussian.png?raw=true)
![logistic_NICE](https://github.com/yhy258/Flow-based-Pytorch/blob/master/images/NICE2_logistic.png?raw=true)
  
EPOCH : 50 train 후 sampling 결과. 왼쪽 이미지는 Gaussian distribution을 prior로 둔 경우이고, 오른쪽은 Logistic Distribution을 prior로 둔 경우이다.  
논문에 명시 된 대로 logistic distribution을 prior로 뒀을 때 더 좋은 결과를 내놓았다.  
  
flow based GM 중 시작점이 되는 NICE라는 모델은 굉장히 단순한 메커니즘을 기반으로 했다. 특히 coupling layer를 알맞게 구성하여 flow based method가 성립되도록 했을 때 좀 신기했다.  
  
  
## Glow
paper : https://proceedings.neurips.cc/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf
Model : https://github.com/yhy258/Flow-based-Pytorch/blob/master/glow.py  
  
설명 : https://deepseow.tistory.com/47  
  
후기 : ![Glow output](https://github.com/yhy258/Flow-based-Pytorch/blob/master/images/glow.png?raw=true)  
  
EPOCH : 50 epoch으로 train한 후 sampling 결과. 하이퍼 파라미터는 오피셜과 동일하게 했지만 왜 이렇게 결과가 안나오는지는 잘 모르겠음..  
-> 훈련이 부족해서인듯 싶다. paper에서도 1400 에폭정도 돌렸을 때 NLL이 수렴함을 보였고, 어떤 pytorch 구현체는 200000 에폭을 돌렸을때 좋은 결과를 보여줬다. 하지만 나는 자원이 부족해 그렇게 돌리기가 쉽지않아 우선 여기에서 마무리하겠음.  
permutation 작업을 1x1 invertible convolution으로 대체 했다는 점이 흥미로웠다.  

# GPU settings

## 압축풀기
tar xvzf cudnn-11.0-linux-x64-v8.0.5.39.tgz

## copy files
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

## Check
ldconfig -N -v $(sed 's/://' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn

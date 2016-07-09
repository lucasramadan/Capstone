# Model Nomenclature

**Note:** All models trained with 19 spaced verbose data, *unless* otherwise stated
* Data has a feature space size of 912
* Labels have 7 possible classes (NOT 8)
  * Dropped 'C' class, because these are really "unknown" structure assignments from DSSP

## Feed-Forward Networks

### FF-2L-500-SIG

##### Code:

```
# data dimensions
n_input, n_ouput = 912, 7

# instantiate model
model_1L = Sequential()

# first layer 
model_1L.add(Dense(output_dim=500, input_dim=n_input))
model_1L.add(Activation('sigmoid'))

# second layer
model_1L.add(Dense(input_dim=500, output_dim=n_output))
model_1L.add(Activation("softmax"))
```

##### Architecture:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
dense_3 (Dense)                  (None, 500)           456500      dense_input_2[0][0]              
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 500)           0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 7)             3507        activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 7)             0           dense_4[0][0]                    
====================================================================================================
Total params: 460007
____________________________________________________________________________________________________

```

##### Performance:
```
Epoch 1/5
161412/161412 [======================] - 84s - loss: 1.2147 - acc: 0.5545 - val_loss: 1.1925 - val_acc: 0.5636
Epoch 2/5
161412/161412 [======================] - 102s - loss: 1.1774 - acc: 0.5665 - val_loss: 1.1573 - val_acc: 0.5746
Epoch 3/5
161412/161412 [======================] - 102s - loss: 1.1472 - acc: 0.5780 - val_loss: 1.1416 - val_acc: 0.5798
Epoch 4/5
161412/161412 [======================] - 103s - loss: 1.0881 - acc: 0.6041 - val_loss: 1.0763 - val_acc: 0.6130
Epoch 5/5
161412/161412 [======================] - 102s - loss: 1.0051 - acc: 0.6430 - val_loss: 1.0175 - val_acc: 0.6456
```

##### Summary:
> Likely could have benefitted from more epochs of training, but decent performance for a simple benchmark network

------- 

### FF-2L-500-ELU-BN-DO

##### Code:

```
# data dimensions
n_input, n_ouput = 912, 7

# instantiate model
model_2L = Sequential()

# first layer, 500 nodes, BatchNormalized, ELU and Dropout
model_2L.add(Dense(output_dim=500, input_dim=n_input))
model_2L.add(BatchNormalization())
model_2L.add(ELU(alpha=0.9))
model_2L.add(Dropout(0.5))

# second layer, 7 nodes, BatchNormalized, SoftMax
model_2L.add(Dense(input_dim=500, output_dim=n_output))
model_2L.add(BatchNormalization())
model_2L.add(Activation("softmax"))
```

##### Architecture: 

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
dense_1 (Dense)                  (None, 500)           456500      dense_input_1[0][0]              
____________________________________________________________________________________________________
batchnormalization_1 (BatchNormal(None, 500)           1000        dense_1[0][0]                    
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 500)           0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 500)           0           elu_1[0][0]                      
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 8)             4008        dropout_1[0][0]                  
____________________________________________________________________________________________________
batchnormalization_2 (BatchNormal(None, 8)             16          dense_2[0][0]                    
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 8)             0           batchnormalization_2[0][0]       
====================================================================================================
Total params: 461524
____________________________________________________________________________________________________

```

##### Performance: 


```
Epoch 1/5
161412/161412 [==============================] - 84s - loss: 1.2625 - acc: 0.5514    
Epoch 2/5
161412/161412 [==============================] - 101s - loss: 1.1485 - acc: 0.5788   
Epoch 3/5
161412/161412 [==============================] - 103s - loss: 1.1006 - acc: 0.6012   
Epoch 4/5
161412/161412 [==============================] - 101s - loss: 1.0464 - acc: 0.6269   
Epoch 5/5
161412/161412 [==============================] - 102s - loss: 0.9947 - acc: 0.6508 
```

##### Summary:
> Training improvements over FF-2L-500-S, due to BatchNormalization, but will likely benefit from complexity increases in architecture. 

----- 

### FF-3L-1000-ELU-BN-DO

##### Code:

```
# data dimensions
n_input, n_ouput = 912, 7

# instantiate model
model_3L = Sequential()

# first layer, 1000 nodes, BatchNormalized, ELU and Dropout
model_3L.add(Dense(input_dim=n_input, output_dim=1000))
model_3L.add(BatchNormalization())
model_3L.add(ELU(alpha=0.9))
model_3L.add(Dropout(0.5))

# second layer, 1000 nodes, BatchNormalized, ELU and Dropout
model_3L.add(Dense(input_dim=1000, output_dim=1000))
model_3L.add(BatchNormalization())
model_3L.add(ELU(alpha=0.9))
model_3L.add(Dropout(0.5))

# third layer, 7 nodes, BatchNormalized, SoftMax
model_3L.add(Dense(input_dim=1000, output_dim=n_output))
model_3L.add(BatchNormalization())
model_3L.add(Activation("softmax"))
```

##### Architecture:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
dense_11 (Dense)                 (None, 1000)          913000      dense_input_6[0][0]              
____________________________________________________________________________________________________
batchnormalization_7 (BatchNormal(None, 1000)          2000        dense_11[0][0]                   
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 1000)          0           batchnormalization_7[0][0]       
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 1000)          0           elu_6[0][0]                      
____________________________________________________________________________________________________
dense_12 (Dense)                 (None, 1000)          1001000     dropout_6[0][0]                  
____________________________________________________________________________________________________
batchnormalization_8 (BatchNormal(None, 1000)          2000        dense_12[0][0]                   
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 1000)          0           batchnormalization_8[0][0]       
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 1000)          0           elu_7[0][0]                      
____________________________________________________________________________________________________
dense_13 (Dense)                 (None, 7)             7007        dropout_7[0][0]                  
____________________________________________________________________________________________________
batchnormalization_9 (BatchNormal(None, 7)             14          dense_13[0][0]                   
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 7)             0           batchnormalization_9[0][0]       
====================================================================================================
Total params: 1925021
____________________________________________________________________________________________________
```

##### Performance:

```
Epoch 1/20
161412/161412 [=====================] - 252s - loss: 1.2438 - acc: 0.5533 - val_loss: 1.1528 - val_acc: 0.5773
Epoch 2/20
161412/161412 [=====================] - 250s - loss: 1.1478 - acc: 0.5779 - val_loss: 1.1134 - val_acc: 0.5935
Epoch 3/20
161412/161412 [=====================] - 252s - loss: 1.0997 - acc: 0.6008 - val_loss: 1.0494 - val_acc: 0.6208
Epoch 4/20
161412/161412 [=====================] - 248s - loss: 1.0282 - acc: 0.6333 - val_loss: 0.9593 - val_acc: 0.6646
Epoch 5/20
161412/161412 [=====================] - 247s - loss: 0.9598 - acc: 0.6637 - val_loss: 0.8864 - val_acc: 0.6961
Epoch 6/20
161412/161412 [=====================] - 250s - loss: 0.9007 - acc: 0.6893 - val_loss: 0.8202 - val_acc: 0.7248
Epoch 7/20
161412/161412 [=====================] - 248s - loss: 0.8538 - acc: 0.7066 - val_loss: 0.7751 - val_acc: 0.7426
Epoch 8/20
161412/161412 [=====================] - 248s - loss: 0.8171 - acc: 0.7217 - val_loss: 0.7428 - val_acc: 0.7584
Epoch 9/20
161412/161412 [=====================] - 249s - loss: 0.7805 - acc: 0.7362 - val_loss: 0.7251 - val_acc: 0.7635
Epoch 10/20
161412/161412 [=====================] - 252s - loss: 0.7545 - acc: 0.7459 - val_loss: 0.6938 - val_acc: 0.7752
Epoch 11/20
161412/161412 [=====================] - 251s - loss: 0.7304 - acc: 0.7559 - val_loss: 0.6650 - val_acc: 0.7885
Epoch 12/20
161412/161412 [=====================] - 249s - loss: 0.7089 - acc: 0.7634 - val_loss: 0.6470 - val_acc: 0.7938
Epoch 13/20
161412/161412 [=====================] - 252s - loss: 0.6893 - acc: 0.7710 - val_loss: 0.6411 - val_acc: 0.7959
Epoch 14/20
161412/161412 [=====================] - 254s - loss: 0.6687 - acc: 0.7776 - val_loss: 0.6296 - val_acc: 0.8015
Epoch 15/20
161412/161412 [=====================] - 254s - loss: 0.6525 - acc: 0.7834 - val_loss: 0.6181 - val_acc: 0.8078
Epoch 16/20
161412/161412 [=====================] - 250s - loss: 0.6384 - acc: 0.7883 - val_loss: 0.5989 - val_acc: 0.8117
Epoch 17/20
161412/161412 [=====================] - 249s - loss: 0.6239 - acc: 0.7940 - val_loss: 0.5887 - val_acc: 0.8164
Epoch 18/20
161412/161412 [=====================] - 250s - loss: 0.6122 - acc: 0.7976 - val_loss: 0.5814 - val_acc: 0.8208
Epoch 19/20
161412/161412 [=====================] - 249s - loss: 0.5990 - acc: 0.8036 - val_loss: 0.5758 - val_acc: 0.8230
Epoch 20/20
161412/161412 [=====================] - 250s - loss: 0.5887 - acc: 0.8068 - val_loss: 0.5677 - val_acc: 0.8266
```

##### Summary:
> Addition of the hidden layer making a significant difference to performance. Aggresive Dropout rate likely the cause of validation accuracy surpassing training accuracy.

-----

### FF-3L-2000-BN-SReLU-DO

##### Code:

```
# data dimensions
n_input, n_ouput = 912, 7

# instantiate model
model_3LW = Sequential()

# first layer, 2000 nodes, BatchNormalized, SReLU and Dropout
model_3LW.add(Dense(input_dim=n_input, output_dim=2000))
model_3LW.add(BatchNormalization())
model_3LW.add(SReLU())
model_3LW.add(Dropout(0.5))

# second layer, 2000 nodes, BatchNormalized, SReLU and Dropout
model_3LW.add(Dense(input_dim=2000, output_dim=2000))
model_3LW.add(BatchNormalization())
model_3LW.add(SReLU())
model_3LW.add(Dropout(0.5))

# third layer, 7 nodes, BatchNormalized, SoftMax
model_3LW.add(Dense(input_dim=2000, output_dim=n_output))
model_3LW.add(BatchNormalization())
model_3LW.add(Activation("softmax"))
```

##### Architecture:

```
____________________________________________________________________________________________________
Layer (type)                       Output Shape        Param #     Connected to                     
====================================================================================================
dense_3 (Dense)                    (None, 2000)        1826000     dense_input_2[0][0]              
____________________________________________________________________________________________________
batchnormalization_3 (BatchNormaliz(None, 2000)        4000        dense_3[0][0]                    
____________________________________________________________________________________________________
srelu_3 (SReLU)                    (None, 2000)        8000        batchnormalization_3[0][0]       
____________________________________________________________________________________________________
dropout_3 (Dropout)                (None, 2000)        0           srelu_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                    (None, 2000)        4002000     dropout_3[0][0]                  
____________________________________________________________________________________________________
batchnormalization_4 (BatchNormaliz(None, 2000)        4000        dense_4[0][0]                    
____________________________________________________________________________________________________
srelu_4 (SReLU)                    (None, 2000)        8000        batchnormalization_4[0][0]       
____________________________________________________________________________________________________
dropout_4 (Dropout)                (None, 2000)        0           srelu_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                    (None, 7)           14007       dropout_4[0][0]                  
____________________________________________________________________________________________________
batchnormalization_5 (BatchNormaliz(None, 7)           14          dense_5[0][0]                    
____________________________________________________________________________________________________
activation_1 (Activation)          (None, 7)           0           batchnormalization_5[0][0]       
====================================================================================================
Total params: 5866021
____________________________________________________________________________________________________
```

##### Performance:

```
Epoch 1/50
161412/161412 [=====================] - 779s - loss: 1.1282 - acc: 0.6094 - val_loss: 0.8211 - val_acc: 0.7298
Epoch 2/50
161412/161412 [=====================] - 848s - loss: 0.8045 - acc: 0.7355 - val_loss: 0.6456 - val_acc: 0.7989
Epoch 3/50
161412/161412 [=====================] - 848s - loss: 0.6554 - acc: 0.7905 - val_loss: 0.5434 - val_acc: 0.8345
Epoch 4/50
161412/161412 [=====================] - 856s - loss: 0.5530 - acc: 0.8247 - val_loss: 0.5200 - val_acc: 0.8439
Epoch 5/50
161412/161412 [=====================] - 854s - loss: 0.4827 - acc: 0.8475 - val_loss: 0.4942 - val_acc: 0.8552
Epoch 6/50
161412/161412 [=====================] - 849s - loss: 0.4257 - acc: 0.8655 - val_loss: 0.4652 - val_acc: 0.8661
Epoch 7/50
161412/161412 [=====================] - 849s - loss: 0.3830 - acc: 0.8796 - val_loss: 0.4563 - val_acc: 0.8735
Epoch 8/50
161412/161412 [=====================] - 850s - loss: 0.3511 - acc: 0.8899 - val_loss: 0.4573 - val_acc: 0.8740
Epoch 9/50
161412/161412 [=====================] - 847s - loss: 0.3270 - acc: 0.8976 - val_loss: 0.4447 - val_acc: 0.8825
Epoch 10/50
161412/161412 [=====================] - 848s - loss: 0.3065 - acc: 0.9034 - val_loss: 0.4570 - val_acc: 0.8824
Epoch 11/50
161412/161412 [=====================] - 852s - loss: 0.2911 - acc: 0.9087 - val_loss: 0.4669 - val_acc: 0.8853
Epoch 12/50
161412/161412 [=====================] - 855s - loss: 0.2767 - acc: 0.9134 - val_loss: 0.4858 - val_acc: 0.8846
Epoch 13/50
161412/161412 [=====================] - 853s - loss: 0.2681 - acc: 0.9155 - val_loss: 0.4574 - val_acc: 0.8882
Epoch 14/50
161412/161412 [=====================] - 854s - loss: 0.2568 - acc: 0.9194 - val_loss: 0.4603 - val_acc: 0.8879
Epoch 15/50
161412/161412 [=====================] - 855s - loss: 0.2465 - acc: 0.9223 - val_loss: 0.4758 - val_acc: 0.8898
Epoch 16/50
161412/161412 [=====================] - 861s - loss: 0.2424 - acc: 0.9243 - val_loss: 0.4719 - val_acc: 0.8893
Epoch 17/50
161412/161412 [=====================] - 860s - loss: 0.2367 - acc: 0.9261 - val_loss: 0.4729 - val_acc: 0.8909
Epoch 18/50
161412/161412 [=====================] - 858s - loss: 0.2299 - acc: 0.9279 - val_loss: 0.4843 - val_acc: 0.8919
Epoch 19/50
161412/161412 [=====================] - 855s - loss: 0.2207 - acc: 0.9309 - val_loss: 0.5043 - val_acc: 0.8897
Epoch 20/50
161412/161412 [=====================] - 853s - loss: 0.2223 - acc: 0.9302 - val_loss: 0.4889 - val_acc: 0.8901
Epoch 21/50
161412/161412 [=====================] - 859s - loss: 0.2145 - acc: 0.9327 - val_loss: 0.4994 - val_acc: 0.8908
Epoch 22/50
161412/161412 [=====================] - 861s - loss: 0.2118 - acc: 0.9336 - val_loss: 0.4883 - val_acc: 0.8941
Epoch 23/50
161412/161412 [=====================] - 864s - loss: 0.2087 - acc: 0.9347 - val_loss: 0.4884 - val_acc: 0.8941
Epoch 24/50
161412/161412 [=====================] - 863s - loss: 0.2035 - acc: 0.9367 - val_loss: 0.4943 - val_acc: 0.8941
Epoch 25/50
161412/161412 [=====================] - 874s - loss: 0.2019 - acc: 0.9375 - val_loss: 0.4816 - val_acc: 0.8945
Epoch 26/50
161412/161412 [=====================] - 876s - loss: 0.1951 - acc: 0.9389 - val_loss: 0.5080 - val_acc: 0.8937
Epoch 27/50
161412/161412 [=====================] - 871s - loss: 0.1949 - acc: 0.9390 - val_loss: 0.5150 - val_acc: 0.8922
Epoch 28/50
161412/161412 [=====================] - 873s - loss: 0.1923 - acc: 0.9408 - val_loss: 0.5048 - val_acc: 0.8936
Epoch 29/50
161412/161412 [=====================] - 876s - loss: 0.1927 - acc: 0.9398 - val_loss: 0.5058 - val_acc: 0.8933
Epoch 30/50
161412/161412 [=====================] - 871s - loss: 0.1894 - acc: 0.9410 - val_loss: 0.5137 - val_acc: 0.8921
Epoch 31/50
161412/161412 [=====================] - 874s - loss: 0.1869 - acc: 0.9418 - val_loss: 0.5071 - val_acc: 0.8927
Epoch 32/50
161412/161412 [=====================] - 880s - loss: 0.1849 - acc: 0.9424 - val_loss: 0.5096 - val_acc: 0.8938
Epoch 33/50
161412/161412 [=====================] - 876s - loss: 0.1816 - acc: 0.9443 - val_loss: 0.5195 - val_acc: 0.8934
Epoch 34/50
161412/161412 [=====================] - 881s - loss: 0.1803 - acc: 0.9443 - val_loss: 0.5405 - val_acc: 0.8926
Epoch 35/50
161412/161412 [=====================] - 879s - loss: 0.1784 - acc: 0.9451 - val_loss: 0.4993 - val_acc: 0.8944
Epoch 36/50
161412/161412 [=====================] - 883s - loss: 0.1789 - acc: 0.9450 - val_loss: 0.4950 - val_acc: 0.8953
Epoch 37/50
161412/161412 [=====================] - 883s - loss: 0.1766 - acc: 0.9451 - val_loss: 0.5060 - val_acc: 0.8932
Epoch 38/50
161412/161412 [=====================] - 886s - loss: 0.1747 - acc: 0.9465 - val_loss: 0.5456 - val_acc: 0.8940
Epoch 39/50
161412/161412 [=====================] - 883s - loss: 0.1734 - acc: 0.9464 - val_loss: 0.5014 - val_acc: 0.8959
Epoch 40/50
161412/161412 [=====================] - 883s - loss: 0.1726 - acc: 0.9474 - val_loss: 0.5032 - val_acc: 0.8955
Epoch 41/50
161412/161412 [=====================] - 881s - loss: 0.1721 - acc: 0.9472 - val_loss: 0.4930 - val_acc: 0.8962
Epoch 42/50
161412/161412 [=====================] - 881s - loss: 0.1694 - acc: 0.9481 - val_loss: 0.5178 - val_acc: 0.8951
Epoch 43/50
161412/161412 [=====================] - 886s - loss: 0.1684 - acc: 0.9476 - val_loss: 0.5370 - val_acc: 0.8954
Epoch 44/50
161412/161412 [=====================] - 880s - loss: 0.1670 - acc: 0.9487 - val_loss: 0.5283 - val_acc: 0.8939
Epoch 45/50
161412/161412 [=====================] - 880s - loss: 0.1676 - acc: 0.9486 - val_loss: 0.5338 - val_acc: 0.8956
Epoch 46/50
161412/161412 [=====================] - 883s - loss: 0.1660 - acc: 0.9488 - val_loss: 0.5282 - val_acc: 0.8948
Epoch 47/50
161412/161412 [=====================] - 887s - loss: 0.1657 - acc: 0.9491 - val_loss: 0.5244 - val_acc: 0.8956
Epoch 48/50
161412/161412 [=====================] - 885s - loss: 0.1653 - acc: 0.9492 - val_loss: 0.5210 - val_acc: 0.8953
Epoch 49/50
161412/161412 [=====================] - 871s - loss: 0.1645 - acc: 0.9495 - val_loss: 0.5423 - val_acc: 0.8948
Epoch 50/50
161412/161412 [=====================] - 876s - loss: 0.1627 - acc: 0.9497 - val_loss: 0.5247 - val_acc: 0.8953
```

##### Summary:
> 

-----

### FF

##### Code:

```

```

##### Architecture:

```

```

##### Performance:

```

```

##### Summary:
> 

-----

### FF

##### Code:

```

```

##### Architecture:

```

```

##### Performance:

```

```

##### Summary:
> 

-----

### FF

##### Code:

```

```

##### Architecture:

```

```

##### Performance:

```

```

##### Summary:
> 

-----

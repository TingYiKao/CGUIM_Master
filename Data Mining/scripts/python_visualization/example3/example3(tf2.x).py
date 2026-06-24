#載入numpy函式庫(套件)
import numpy as np 
#載入cifar10類別
from tensorflow.keras.datasets import cifar10 
#載入keras 的序列式模型類別
from tensorflow.keras.models import Sequential 
#載入Dense, Flatten, Conv2D, MaxPooling2D, Dropout等類別
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout 
#載入to_categorical函式
from tensorflow.keras.utils import to_categorical 

#1-0 認識Cifar10彩色圖片資料集
print("#1-0 認識Cifar10彩色圖片資料集")

#指定亂數種子
seed = 5811          #亂數種子, 可以改用其他數字
np.random.seed(seed) #啟動亂數種子

#載入 Cifar10 資料集, 如果是第一次載入會自行下載資料集
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

#顯示訓練和測試資料集的形狀
print("X_train.shape: ", X_train.shape)  #訓練資料集的輸入形狀
print("Y_train.shape: ", Y_train.shape)  #訓練資料集的標籤資料
print("X_test.shape: ", X_test.shape)    #測試資料集的輸入形狀
print("Y_test.shape: ", Y_test.shape)    #測試資料集的標籤資料

#顯示訓練資料集中的第1張圖片資料
print(X_train[0])   # 像素值資料
print(Y_train[0])   # 標籤資料

#使用Matplotlib顯示青蛙的彩色圖片
import matplotlib.pyplot as plt        #滙入matplotlib.pyplot函數
plt.imshow(X_train[0], cmap="binary")  #顯示X_train[0],  colormap使用binary
plt.title("Label: " + str(Y_train[0])) #設定圖表標題
plt.axis("off") #不顯示座標
plt.show()      #顯示圖片

#繪出12張圖片
for i in range(0, 12):      
    ax = plt.subplot(4,3,i+1)       #設定繪製位置
    ax.imshow(X_train[i], cmap="binary") #顯示X_train[i], colormap使用binary
    ax.set_title("Label: " + str(Y_train[i])) #設定圖表標題
    ax.axis("off")                        #不顯示座標
plt.subplots_adjust(hspace = .5) #圖片之間的行距
plt.show()  #顯示圖片

#因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

#One-hot編碼
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

#3.2 步驟二：定義模型
print("#3-2 步驟二：定義模型")
#建立一個序列模型
model = Sequential()                                          
#Conv2D(卷積層), 有32個大小為3*3的filter(過濾器)，保留原圖片大小, 輸入的資料為(32*32*3)，啟動函數為relu
model.add(Conv2D(32, kernel_size=(3, 3), padding="same",      
                 input_shape=(32,32,3), activation="relu"))  
#MaxPooling2D(池化層)
model.add(MaxPooling2D(pool_size=(2, 2)))                     
#Dropout(丟棄層) 隨機放棄25%的神經元
model.add(Dropout(0.25))                                      
#Conv2D(卷積層), 64個大小為3*3的filter(過濾器)，保留原圖片大小, 輸入的資料為(32*32*3)，啟動函數為relu
model.add(Conv2D(64, kernel_size=(3, 3),                     
                   padding="same", activation="relu"))        
#MaxPooling2D(池化層)
model.add(MaxPooling2D(pool_size=(2, 2)))
#Dropout(丟棄層) 隨機放棄25%的神經元                     
model.add(Dropout(0.25))                                      
#Flattening(展平層) 將高維陣列轉為1維
model.add(Flatten())                                          
#Dense(密集層)，有512個神經元，啟動函數為relu
model.add(Dense(512, activation="relu"))                      
#Dropout(丟棄層), 隨機放棄40%的神經元
model.add(Dropout(0.4))                                       
#Dense(密集層)，有10個輸出神經元(因為有10類)，啟動函數為softmax
model.add(Dense(10, activation="softmax"))                    


#顯示模型摘要資訊
model.summary()   

#3-3 步驟三：編譯模型
print("#3-3 步驟三：編譯模型")
#損失函數為categorical_crossentropy
#最佳化方式採用Adam
#評估指標使用accuracy
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])     
			  
#3-4 步驟四：訓練模型
print("#3-4 步驟四：訓練模型")
#X_train：訓練資料的特徵資料
#Y_train：訓練資料的輸出欄位
#validation_split：分割出驗證資料集的比例
#epochs：訓練週期
#batch_size：批次大小
#verbose：訓練過程中訊息顯示的詳細程度
history = model.fit(X_train, Y_train, validation_split=0.2, 
                    epochs=9, batch_size=128, verbose=2)
					
#3-5 步驟五：評估與儲存模型
print("#3-5 步驟五：評估與儲存模型")
#計算訓練資料集的準確度 
loss, accuracy = model.evaluate(X_train, Y_train)       
print("訓練資料集的準確度 = %.2f" % (accuracy))
#計算測試資料集的準確度 
loss, accuracy = model.evaluate(X_test, Y_test)          
print("測試資料集的準確度 = %.2f" % (accuracy))
#儲存Keras模型
print("Saving Model: cifar10.h5 ...")
model.save("cifar10.h5")

#顯示圖表來分析模型的訓練過程
#滙入matplotlib.pyplot類別
import matplotlib.pyplot as plt 

#顯示訓練和驗證損失
#取得訓練過程中訓練資料的每一個週期損失函數值
loss = history.history["loss"]         
#設定週期範圍
epochs = range(1, len(loss)+1)         
#取得訓練過程中驗證資料的每一個週期損失函數值
val_loss = history.history["val_loss"] 
#驗證資料的每一個週期損失函數值
#X座標:epochs， y座標:loss, 線的型態為bo-，圖標為Training Loss
plt.plot(epochs, loss, "bo-", label="Training Loss") 
#X座標:epochs， y座標:loss, 線的型態為ro-，圖標為Validation Loss
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
#設定圖片的標題
plt.title("Training and Validation Loss")
#設定圖片的X軸標題 
plt.xlabel("Epochs")    
#設定圖片的Y軸標題
plt.ylabel("Loss")      
#要顯示座標軸上的刻度、數字
plt.legend()            
#將圖片顯示出來
plt.show()              

#顯示訓練和驗證準確率
#取得訓練過程中訓練資料的每一個週期準確率
acc = history.history["accuracy"]    
#設定週期範圍
epochs = range(1, len(acc)+1)   
#取得訓練過程中驗證資料的每一個週期準確率
val_acc = history.history["val_accuracy"]   
#X座標:epochs， y座標:loss, 線的型態為bo-，圖標為Training Acc
plt.plot(epochs, acc, "bo-", label="Training Acc")
#X座標:epochs， y座標:loss, 線的型態為ro-，圖標為Validation Acc
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
#設定圖片的標題
plt.title("Training and Validation Accuracy")
#設定圖片的X軸標題
plt.xlabel("Epochs")    
#設定圖片的Y軸標題
plt.ylabel("Accuracy")  
#要顯示座標軸上的刻度、數字
plt.legend()     
#將圖片顯示出來       
plt.show()

#3.6 載入模型並使用混淆矩陣分析預測結果
print("3-6 載入模型並使用混淆矩陣分析預測結果")
#載入pandas函數庫(套件)
import pandas as pd
#載入load_model函數
from tensorflow.keras.models import load_model 

#載入資料集
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
#因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_test = X_test.astype("float32") / 255
#備份 Y_test 資料集
Y_test_bk = Y_test.copy()       
#將標籤資料改用one-hot編碼
Y_test = to_categorical(Y_test)  

#建立Keras的Sequential模型
#建立一個序列模型
model = Sequential()   
#載入之前建立的模型
model = load_model("cifar10.h5") 
#編譯模型
 #損失函數為categorical_crossentropy
 #最佳化方式採用Adam
 #評估指標使用accuracy
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 評估模型
print("Testing ...")
#計算測試資料集的準確度 
loss, accuracy = model.evaluate(X_test, Y_test) 
print("測試資料集的準確度 = %.2f" % (accuracy))
# 計算分類的預測值
print("\nPredicting ...")
#使用模型來預測測試資料的類別 
Y_pred = model.predict_classes(X_test)           
#顯示混淆矩陣
#crosstab的第1個參數為列的資料，第2個參數為欄的資料, rownames為列的標籤, colnames為欄的標籤
tb = pd.crosstab(Y_test_bk.astype(int).flatten(), 
                 Y_pred.astype(int),
                 rownames=["label"], colnames=["predict"])
print(tb)  #顯示混淆矩陣
tb.to_html("example3_1.html") #儲存成HTML檔案

#3.7 繪出某1張圖片之0-9分類的預測機率
print("#3.7 繪出某1張圖片之0-9分類的預測機率")
#選一個測試的圖片 
i = 8  
#取出編號為8的圖片資料，為1個3D張量
img = X_test[i]   
#將圖片轉換成4D張量
X_test_img = img.reshape(1, 32, 32, 3).astype("float32") 
#正規化
X_test_img = X_test_img / 255  
#建立一個序列模型
model = Sequential() 
#載入之前建立的模型
model = load_model("cifar10.h5") 
#編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 

#繪出圖表的預測結果
plt.figure() 
#子圖在2列1欄中的第1列(在上面的圖)
plt.subplot(2,1,1)  
#設定圖標
plt.title("Example of Image:" + str(Y_test[i])) 
#顯示圖片 
plt.imshow(img, cmap="binary")                  
#不顯示座標軸
plt.axis("off")                                 
#預測結果的機率
probs = model.predict(X_test_img, batch_size=1)
#子圖在2列1欄中的第2列(在下面的圖)
plt.subplot(2,1,2)
#設定圖標
plt.title("Probabilities for Each Image Class")   
#繪製長條圖
plt.bar(np.arange(10), probs.reshape(10), align="center")
#顯示x軸上的標記數字
plt.xticks(np.arange(10),np.arange(10).astype(str))       
#將圖繪製出來
plt.show()


from tensorflow.keras.datasets import mnist #載入 mnist 資料集, 如果是第一次載入會自行下載資料集
#載入 mnist 資料集, 如果是第一次載入會自行下載資料集
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
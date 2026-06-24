import numpy as np #載入numpy函式庫(套件)  
import pandas as pd  #滙入pandas函式庫
import tensorflow as tf #滙入tensorflow 函式庫
from tensorflow import keras #從tensorflow 中滙入keras
from tensorflow.keras import layers  #從tensorflow.keras 中滙入layers

#2-1 載入波士頓房屋資料集
print("#2-1 載入波士頓房屋資料集")
df = pd.read_csv("./boston_housing.csv")
#查看前5筆記錄
print(df.head())  
#將前5筆記錄儲存成html格式的檔案
df.head().to_html("./example2_1.html") 

#顯示資料集的形狀
print(df.shape)

#2-2 探索資料
print("#2-2 探索資料")
#顯示資料集的描述
print(df.describe())
#將描述資料儲存成html格式的檔案
df.describe().to_html("./example2_2.html") 

#2-3 資料視覺化
print("#2-3 資料視覺化")
import seaborn as sns
#減少一些變數，顯示兩兩變變之間的關係 
#axis=1表示刪除欄; axis=0表示刪除列。
df2=df.drop(["tax","b","nox","chas","crim","ptratio","age","zn"],axis=1)
sns.pairplot(df2)  #將兩兩變數的圖繪製出來

#2-4 K-fold交叉驗證
print("#2-4 K-fold交叉驗證")
#2-4-1 載入和打亂資料集
print("#2-4-1 載入和打亂資料集")
seed=5811             #指定亂數種子
np.random.seed(seed)  #啟動亂數
df = pd.read_csv("./boston_housing.csv")
#儲存讀入df的Numpy陣列，不含欄位名稱 
dataset = df.values         
#使用亂數打亂df每筆資料的先後順序
np.random.shuffle(dataset)  

#2-4-2 分割資料集
print("#2-4-2 分割資料集")
#分割成特徵資料和標籤資料
#取0-13的欄位資料，共13個欄位的值(輸入資料)
X = dataset[:, 0:13] 
#取編號為13的欄位資料: 第14個欄位的值(輸出資料；自住房屋的中位數價格) 
Y = dataset[:, 13]  
 
#特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)

#分割訓練和測試資料集
#訓練資料前404筆
X_train, Y_train = X[:404], Y[:404]     
#測試資料後102筆
X_test, Y_test = X[404:], Y[404:]       

#2-4-3 定義模型與編譯模型
print("#2-4-3 定義模型與編譯模型")
#定義第1個模型
def build_model():
    #建立一個序列模型
    model = keras.models.Sequential()  #建立一個序列模型
	#Dense(密集層)，有11個神經元，輸入欄位數為X_train的欄位數, 啟動函數為relu
    model.add(layers.Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
	#Dense(密集層)，有1個輸出神經元
    model.add(layers.Dense(1))
    #編譯模型
	  #損失函數為"mse
      #最佳化方式採用Adam
      #評估指標使用mae
    model.compile(loss="mse", optimizer="adam", 
                  metrics=["mae"])
    return model

#定義第2個模型
def build_new_model():
    #建立一個序列模型
    model = keras.models.Sequential()  #建立一個序列模型
	#Dense(密集層)，有32個神經元，輸入欄位數為X_train的欄位數, 啟動函數為relu
    model.add(layers.Dense(32, input_shape=(X_train.shape[1],), activation="relu")) 
	#Dense(密集層)，有16個神經元, 啟動函數為relu	
    model.add(layers.Dense(16, activation="relu"))
	#Dense(密集層)，有1個輸出神經元
    model.add(layers.Dense(1))
    #編譯模型
	  #損失函數為"mse
      #最佳化方式採用Adam
      #評估指標使用mae	
    model.compile(loss="mse", optimizer="adam", 
                  metrics=["mae"])
    return model

#2-4-4 訓練模型與模型評估
print("#2-4-4 訓練模型與模型評估")
print("第1個model")
k = 4
nb_val_samples = len(X_train) // k
nb_epochs = 80
mse_scores = []
mae_scores = []
for i in range(k):
    print("Processing Fold #" + str(i))
    # 取出驗證資料集
    X_val = X_train[i*nb_val_samples: (i+1)*nb_val_samples]
    Y_val = Y_train[i*nb_val_samples: (i+1)*nb_val_samples]
    # 結合出訓練資料集
    X_train_p = np.concatenate(
            [X_train[:i*nb_val_samples],
            X_train[(i+1)*nb_val_samples:]], axis=0)
    Y_train_p = np.concatenate(
            [Y_train[:i*nb_val_samples],
            Y_train[(i+1)*nb_val_samples:]], axis=0)
    model = build_model()
    # 訓練模型
	  #X_train_p：訓練資料的特徵資料
      #Y_train_p：訓練資料的輸出欄位
      #epochs：訓練週期
      #batch_size：批次大小
      #verbose：訓練過程中訊息顯示的詳細程度
      #註: 沒有使用validation_split參數
    model.fit(X_train_p, Y_train_p, epochs=nb_epochs, 
              batch_size=16, verbose=0)
    # 評估模型
    mse, mae = model.evaluate(X_val, Y_val)
    mse_scores.append(mse)
    mae_scores.append(mae)

#顯示驗證資料的平圴MSE和MAE    
print("MSE_val: ", np.mean(mse_scores))
print("MAE_val: ", np.mean(mae_scores))

#使用測試資料評估模型
mse, mae = model.evaluate(X_test, Y_test)  
#顯示測試資料的平圴MSE和MAE   
print("MSE_test: ", mse)
print("MAE_test: ", mae)

print("第2個model")
k = 4
nb_val_samples = len(X_train) // k
nb_epochs = 80
mse_scores = []
mae_scores = []
for i in range(k):
    print("Processing Fold #" + str(i))
    # 取出驗證資料集
    X_val = X_train[i*nb_val_samples: (i+1)*nb_val_samples]
    Y_val = Y_train[i*nb_val_samples: (i+1)*nb_val_samples]
    # 結合出訓練資料集
    X_train_p = np.concatenate(
            [X_train[:i*nb_val_samples],
            X_train[(i+1)*nb_val_samples:]], axis=0)
    Y_train_p = np.concatenate(
            [Y_train[:i*nb_val_samples],
            Y_train[(i+1)*nb_val_samples:]], axis=0)
    model = build_new_model()
    # 訓練模型
	  #X_train_p：訓練資料的特徵資料
      #Y_train_p：訓練資料的輸出欄位
      #epochs：訓練週期
      #batch_size：批次大小
      #verbose：訓練過程中訊息顯示的詳細程度
      #註: 沒有使用validation_split參數
    model.fit(X_train_p, Y_train_p, epochs=nb_epochs, 
              batch_size=16, verbose=0)
    # 評估模型
    mse, mae = model.evaluate(X_val, Y_val)
    mse_scores.append(mse)
    mae_scores.append(mae)
    
#顯示驗證資料的平圴MSE和MAE    
print("MSE_val: ", np.mean(mse_scores))
print("MAE_val: ", np.mean(mae_scores))

# 使用測試資料評估模型
mse, mae = model.evaluate(X_test, Y_test)    
#顯示測試資料的平圴MSE和MAE 
print("MSE_test: ", mse)
print("MAE_test: ", mae)

#假設最好的模型
print("假設最好的模型")
#建立一個序列模型
model = keras.models.Sequential()  #建立一個序列模型
#Dense(密集層)，有32個神經元，輸入欄位數為X_train的欄位數, 啟動函數為relu
model.add(layers.Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
#Dense(密集層)，有32個神經元，啟動函數為relu
model.add(layers.Dense(32, activation="relu"))
#Dense(密集層)，有1個輸出神經元
model.add(layers.Dense(1))

#編譯模型
  #損失函數為"mse
  #最佳化方式採用Adam
  #評估指標使用mae	
model.compile(loss="mse", optimizer="adam", 
                  metrics=["mae"])

#訓練模型
  #X_train：訓練資料的特徵資料
  #Y_train：訓練資料的輸出欄位
  #epochs：訓練週期
  #batch_size：批次大小
  #verbose：訓練過程中訊息顯示的詳細程度
  #註: 沒有使用validation_split參數
model.fit(X_train, Y_train, epochs=nb_epochs, 
              batch_size=16, verbose=0)

#評估模型
mse, mae = model.evaluate(X_test, Y_test)  
#顯示測試資料的平圴MSE和MAE 
print("MSE_test: ", mse)
print("MAE_test: ", mae)

#2-5 儲存模型
print("#2-5 儲存模型")
#方法一：分開儲存模型結構與權重
#儲存模型結構到檔案中
json_str = model.to_json()
with open("example2_model.config", "w") as text_file:
     text_file.write(json_str)
#儲存模型的權重到檔案中	 
model.save_weights("example2_model.weight")


#方法二：同時儲存模型結構與權重
model.save("example2.h5")


#2-6 模型呼叫
print("#2-6 模型呼叫")

#模型呼叫  方法一： 
#載入model_from_json函數
from tensorflow.keras.models import model_from_json
#建立一個序列模型
model = keras.models.Sequential()  #建立一個序列模型
#讀入模型結構
with open("example2_model.config", "r") as text_file:
    json_str = text_file.read()
model = model_from_json(json_str)
#讀入權重檔
model.load_weights("example2_model.weight", by_name=False)


#模型呼叫  方法二： 
#載入load_model函數
from tensorflow.keras.models import load_model
#建立一個序列模型
model = keras.models.Sequential()  #建立一個序列模型
#載入之前建立的模型
model = load_model("example2.h5")


#2.7預測並繪製實際值與預測值之散佈圖
print("#2.7預測並繪製實際值與預測值之散佈圖")
#滙入matplotlib.pyplot函數
import matplotlib.pyplot as plt

#載入load_model函數
from tensorflow.keras.models import load_model
#建立一個序列模型
model = keras.models.Sequential()  #建立一個序列模型
#載入之前建立的模型
model = load_model("example2.h5")

#編譯模型
  #損失函數為categorical_crossentropy
  #最佳化方式採用Adam
  #評估指標使用accuracy
model.compile(loss="mse", optimizer="adam", 
                  metrics=["mae"])

#使用模型來預測測試資料 		  
Y_pred=model.predict(X_test)
#繪製散佈圖
plt.scatter(Y_test, Y_pred)
#設定X軸的標籤
plt. xlabel("Real Value")
#設定y軸的標籤
plt. ylabel("Predicted Value")
#設定圖片的標題
plt. title("Scatter Graph")
plt.show()




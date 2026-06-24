#滙入pandas函式庫
import pandas as pd

#1-1 載入資料集
print("#2-1 載入資料集")
df = pd.read_csv("./titanic_data.csv")

#顯示資料集的形狀
print(df.shape)

#1-2 探索資料
print("#1-2 探索資料")
# 查看前5筆記錄
print(df.head())
#將前5筆記錄儲存成html格式的檔案
df.head().to_html("example2_1.html") 

#顯示資料集的描述資料
print(df.describe())
#將描述資料儲存成html格式的檔案
df.describe().to_html("example2_2.html")

#1-3 資料視覺化
print("#1-3 資料視覺化")
#滙入seaborn函數庫(套件)
import seaborn as sns 
#減少一些變數，顯示兩兩變變之間的關係
#axis=1表示刪除欄; axis=0表示刪除列。
df2=df.drop(["name","ticket","cabin","parch","sibsp"], axis=1)
#將兩兩變數的圖繪製出來
sns.pairplot(df2, hue="survived")


#1-4 資料預處理
print("#2-4 資料預處理-(1)刪除不需要的欄位")
#資料集中的name、ticket和cabin不是特徵資料，可以用drop函式將這些欄位刪除掉
#axis=1表示刪除欄; axis=0表示刪除列。
df=df.drop(["name", "ticket", "cabin"], axis=1)
print("#1-4 資料預處理-(2)處理遺失資料")
# 顯示資料集的資訊
print(df.info())

# 顯示沒有資料的筆數
print(df.isnull().sum())


print("#1-4 資料預處理-(2)處理遺失資料")
#處理遺失資料
#用平均值來填入遺漏值：
#取得平均年齡
NewValue= df["age"].mean()                                 
#用平均年齡來補遺漏值
df["age"] = df["age"].fillna(value=NewValue)               
#取得平均費用
NewValue = df["fare"].mean()                               
#用平均費用來補遺漏值
df["fare"] = df["fare"].fillna(value=NewValue)             

#用眾數來填入遺漏值
#顯示各個登船的港口代碼的筆數
print(df["embarked"].value_counts())                   
#有最多筆數的登船的港口代碼
NewValue=df["embarked"].value_counts().idxmax()       
#用有最多筆數的登船的港口代碼來補遺漏值 print(NewValue)                                 
df["embarked"] = df["embarked"].fillna(value=NewValue)
#再次顯示各個登船的港口代碼的筆數
print(df["embarked"].value_counts())                  


print("#1-4 資料預處理-(3)轉換分類資料")
#轉換分類資料
#則欄位值為female轉為1，male轉為0
df["sex"] = df["sex"].map({"female": 1, "male": 0}).astype(int) 

#Embarked欄位的One-hot編碼
#新增的欄位都會以有embarked字眼，然後再加上底線以及原代碼
enbarked_one_hot = pd.get_dummies(df["embarked"], prefix="embarked") 
#將原本的embarked欄位去除
df = df.drop("embarked", axis=1) 
#在資料集加上新增的One-hot編碼的多個欄位
df = df.join(enbarked_one_hot)   

#1-4 資料預處理-(4)欄位順序的移動"
print("#1-4 資料預處理-(4)欄位順序的移動")
#將標籤的 survived 欄位移至最後
#將資料集中的survived欄位去除，而去除的survived欄位儲存在df_survived中
df_survived = df.pop("survived")     
#將df_survived欄位加到資料集中
df["survived"] = df_survived         
#顯示資料集的前5筆資料
print(df.head())                     
#顯示資料集的前5筆資料儲存成html格式的檔案
df.head().to_html("example2_3.html") 

#1-4 資料預處理-(5)將資料分割成訓練和測試資料集
print("#1-4 資料預處理-(5)將資料分割成訓練和測試資料集")
import numpy as np
#分割成訓練(80%)和測試(20%)資料集
#設定亂數種子
seed = 5811                            
#啟動亂數種子
np.random.seed(seed)                   
#記錄隨機值小於0.8的資料
mask = np.random.rand(len(df)) < 0.8   
#指定訓練資料
df_train = df[mask]                    
#指定測試資料
df_test = df[~mask]                    
#顯示df_train的形狀
print("Train:", df_train.shape)        
#顯示df_test的形狀
print("Test:", df_test.shape)          

#1-4資料預處理-(6)將資料儲存成訓練和測試的CSV檔案
print("#1-4資料預處理-(6)將資料儲存成訓練和測試的CSV檔案")
#將訓練資料儲存csv檔案，不包含資料的索引(index=False)
df_train.to_csv("titanic_train.csv", index=False)
#將測試資料儲存csv檔案，不包含資料的索引(index=False)
df_test.to_csv("titanic_test.csv", index=False)   

#1-4 資料預處理-(7)將資料分割成特徵與標籤資料
print("#1-4 資料預處理-(7)將資料分割成特徵與標籤資料")
#載入Titanic的訓練和測試資料集
df_train = pd.read_csv("./titanic_train.csv")
df_test = pd.read_csv("./titanic_test.csv")
#讀入df_train DataFrame的Numpy陣列，不含欄位名稱
dataset_train = df_train.values   
#讀入df_testDataFrame的的Numpy陣列，不含欄位名稱
dataset_test = df_test.values     

#分割成特徵資料和標籤資料
#取0-8的欄位資料，共9個欄位的值(輸入資料)
X_train = dataset_train[:, 0:9]   
#取編號為9的欄位資料: 第10個欄位的值(輸出資料；存活的分類標計)
Y_train = dataset_train[:, 9]     
#取0-8的欄位資料，共9個欄位的值(輸入資料)
X_test = dataset_test[:, 0:9]     
#取編號為9的欄位資料: 第10個欄位的值(輸出資料；存活的分類標計)
Y_test = dataset_test[:, 9]       

#特徵標準化
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)


#1.5定義模型
print("#1-5 定義模型")
import tensorflow as tf #滙入tensorflow 函式庫
from tensorflow import keras #從tensorflow 中滙入keras
from tensorflow.keras import layers  #從tensorflow.keras 中滙入layers

#第1種
#建立一個序列模型 
model = tf.keras.models.Sequential([          
#Dense(密集層)，有11個神經元，輸入欄位數為X_train的欄位數, 啟動函數為relu
tf.keras.layers.Dense(11, input_dim=X_train.shape[1], activation=tf.nn.relu),
#Dense(密集層)，有11個神經元，啟動函數為relu
tf.keras.layers.Dense(11, activation=tf.nn.relu),    
#Dense(密集層)，有1個神經元(因為只有2類)，啟動函數為sigmoid
tf.keras.layers.Dense(1, activation=tf.nn.sigmoid) 
])


#第2種
model = keras.models.Sequential()  #建立一個序列模型
model.add(layers.Dense(11, input_dim=X_train.shape[1], activation="relu")) #密集層
model.add(layers. Dense(11, activation="relu")) #密集層
model.add(layers. Dense(1, activation="sigmoid")) #密集層


#顯示模型摘要資訊
model.summary()   

#1-6 編譯模型
print("#1-6 編譯模型")
#損失函數為binary_crossentropy
#最佳化方式採用Adam
#評估指標使用accuracy
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

#1-7訓練模型
print("#1-7訓練模型")
print("Training ...")
#X_train：訓練資料的特徵資料
#Y_train：訓練資料的輸出欄位
#validation_split：分割出驗證資料集的比例
#epochs：訓練週期
#batch_size：批次大小
#verbose：訓練過程中訊息顯示的詳細程度
history = model.fit(X_train, Y_train, validation_split=0.2, 
          epochs=100, batch_size=10)
		  
#1-8 評估模型
print("#1-8 評估模型")
print("\nTesting ...")
#計算訓練資料集的準確度 
loss, accuracy = model.evaluate(X_train, Y_train) 
print("訓練資料集的準確度 = %.4f" % (accuracy))
#計算測試資料集的準確度 
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度 = %.4f" % (accuracy))


#1-9 顯示圖表來分析模型的訓練過程
print("#1-9 顯示圖表來分析模型的訓練過程")
#滙入matplotlib.pyplot類別
import matplotlib.pyplot as plt       

#顯示訓練和驗證損失
#取得訓練過程中訓練資料的每一個週期損失函數值
loss = history.history["loss"]         
#設定週期範圍
epochs = range(1, len(loss)+1)         
#取得訓練過程中驗證資料的每一個週期損失函數值
val_loss = history.history["val_loss"] 
#繪出驗證資料的每一個週期損失函數值
#X座標:epochs， y座標:loss, 線的型態為bo-，圖標為Training Loss
plt.plot(epochs, loss, "bo-", label="Training Loss")
#X座標:epochs， y座標:loss, 線的型態為ro-，圖標為Validation Loss
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss") #設定圖片的標題
#設定圖片的X軸標題
plt.xlabel("Epochs")
#設定圖片的Y軸標題
plt.ylabel("Loss")      
#要顯示座標軸上的刻度、數字
plt.legend()           
#將圖片顯示出來 
plt.show()             

#顯示訓練和驗證準確度
#取得訓練過程中訓練資料的每一個週期準確率
acc = history.history["accuracy"]
#設定週期範圍    
epochs = range(1, len(acc)+1)  
#取得訓練過程中驗證資料的每一個週期準確率
val_acc = history.history["val_accuracy"]
#繪出驗證資料的每一個週期準確率
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


#1-10 使用全部的訓練資料集來訓練模型
print("#1-10 使用全部的訓練資料集來訓練模型")
#訓練模型
print("Training ...")
#X_train：訓練資料的特徵資料
#Y_train：訓練資料的輸出欄位
#validation_split：分割出驗證資料集的比例
#epochs：訓練週期
#batch_size：批次大小
#verbose：訓練過程中訊息顯示的詳細程度
#註: 沒有使用validation_split參數
model.fit(X_train, Y_train, epochs=18, batch_size=10, verbose=0) 
#評估模型
print("\nTesting ...")
#計算訓練資料集的準確度 
loss, accuracy = model.evaluate(X_train, Y_train) 
print("訓練資料集的準確度 = %.4f" % (accuracy))
#計算測試資料集的準確度 
loss, accuracy = model.evaluate(X_test, Y_test)   
print("測試資料集的準確度 = %.4f" % (accuracy))  
# 儲存Keras模型
print("Saving Model: titanic.h5 ...")
model.save("titanic.h5")


#1-11 載入模型、編譯模型和評估模型
print("#1-11 載入模型、編譯模型和評估模型")

from tensorflow import keras
#載入之前建立的模型
model = keras.models.load_model("titanic.h5") 

#編譯模型
#損失函數為binary_crossentropy
#最佳化方式採用Adam
#評估指標使用accuracy
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

#評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度 = %.4f" % (accuracy))

#1-12 預測鐵達尼號乘客是否生存
print("#1-12 預測鐵達尼號乘客是否生存")
#計算分類的預測值
print("\nPredicting ...")
#使用模型來預測測試資料的類別 
Y_pred = model.predict_classes(X_test) 
print(Y_pred[:,0])
print(Y_test.astype(int))

#顯示混淆矩陣
#crosstab的第1個參數為列的資料，第2個參數為欄的資料, rownames為列的標籤, colnames為欄的標籤
tb = pd.crosstab(Y_test.astype(int), Y_pred[:,0],
                 rownames=["label"], colnames=["predict"])
#顯示混淆矩陣
print(tb)
#儲存成HTML檔案
tb.to_html("example2_4.html")

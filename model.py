import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("6HK.csv")

# Chọn các cột cần thiết
cdf = df[['SoTcD_1','SoTcR_1','DTB_1','SoTcD_2','SoTcR_2','DTB_2','SoTcD_3','SoTcR_3','DTB_3','SoTcD_4','SoTcR_4','DTB_4','SoTcD_5','SoTcR_5','DTB_5','SoTcD_6','SoTcR_6','DTB_6','KetQua']]
x = cdf.iloc[:, :18]
y = cdf.iloc[:, -1]

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=150)

# Huấn luyện mô hình rừng ngẫu nhiên vs 50 cây
clf = RandomForestClassifier(n_estimators=50)
clf.fit(x_train, y_train)

# Lưu trữ mô hình đã huấn luyện
pickle.dump(clf, open('model.pkl', 'wb'))
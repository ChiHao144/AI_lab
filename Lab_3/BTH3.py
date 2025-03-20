from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from mglearn.datasets import make_forge
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC


import ssl

# Tắt xác thực SSL
ssl._create_default_https_context = ssl._create_unverified_context


# 1/ Download iris data and use skicit-learn to import. Try to call the attribute data of the variable iris.
print("==========Cau 1==========")
iris_data = load_iris()
print(iris_data.data[:5])  # in ra 5 dong dau tien cua du lieu
print("------Cach 2------")
iris = load_iris()
iris_data = iris.data
print(iris_data[:5])

# 2/ How to know what kind of flower belongs each item? How to know the correspondence between
# the species and the number?
print("==========Cau 2==========")
iris_target = iris.target
print(iris_target)
target_names = iris.target_names
print(target_names)

# 3/ Create a scatter plot that displays three different species in three different colors; X-axis will represent
# the length of the sepal while the y-axis will represent the width of the sepal.
print("==========Cau 3==========")
sepal_length = iris.data[:, 0]  # Chiều dài của đài hoa
sepal_width = iris.data[:, 1]  # Chieu rong cua dai hoa
x = sepal_length[iris.target == 0]
y = sepal_width[iris.target == 0]
plt.scatter(x, y, label="Setosa", color="red")
plt.xlabel("sepal lenght")
plt.ylabel("sepal width")
plt.title("Bieu do phan tan cua cac loai hoa Iris")
plt.legend()
plt.show()

# 4/ Using reduce dimension, here using PCA, create a new dimension (=3, called principle component).
print("==========Cau 4==========")
iris = load_iris()
X = iris.data  # Dữ liệu gốc
y = iris.target  # Nhãn loài hoa
Z = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(Z)
print("Tỷ lệ phương sai giải thích cho từng thành phần chính:", pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.title('Biểu đồ phân tán của các thành phần chính')
plt.colorbar(scatter, label='Loài hoa')
plt.show()

# 5/ Using k-nearest neighbor to classify the group that each species belongs to. First, create a training set
# and test set; with 140 will be used as a training set, and 10 remaining will be used as test set.
print("==========Cau 5==========")
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=140, test_size=10, random_state=42)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_train shape: ", Y_train.shape)
print("Y_test shape: ", Y_test.shape)

# 6/ Next, apply the K-nearest neighbor, try with K=5.
print("==========Cau 6==========")
knn = KNeighborsClassifier(n_neighbors=5)  # Sử dụng 5 láng giềng gần nhất
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
print(f"Predicted lables: {y_pred}")

# 7/ Finally, you can compare the results predicted with the actual observed contained in the y_test.
print("==========Cau 7==========")
accuracy = accuracy_score(Y_test, y_pred)
print("Độ chính xác của mô hình k-NN với k=5:", accuracy)

# 8/ Now, you can visualize all this using decision boundaries in a space represented by the 2D scatterplot of sepals
print("==========Cau 8==========")
# 1. Tải dữ liệu Iris
iris = load_iris()
X = iris.data[:, :2]  # Chọn hai đặc trưng: chiều dài và chiều rộng của đài hoa
y = iris.target  # Nhãn loài hoa
# 2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=140, test_size=10, random_state=42)
# 3. Sử dụng k-NN với k=5 để huấn luyện mô hình
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# 4. Tạo lưới điểm để dự đoán nhãn
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
# 5. Dự đoán nhãn cho các điểm trong lưới
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# 6. Vẽ biểu đồ phân tán và ranh giới quyết định
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')  # Vẽ ranh giới quyết định
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', marker='o', label='Tập huấn luyện')
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', marker='x', label='Tập kiểm tra')
plt.xlabel('Chiều dài của đài hoa (sepal length)')
plt.ylabel('Chiều rộng của đài hoa (sepal width)')
plt.title('Ranh giới quyết định của mô hình k-NN với k=5')
plt.legend()
plt.show()

# 9/ Tải xuống tập dữ liệu bệnh tiểu đường. Để dự đoán mô hình, chúng ta sử dụng hồi quy tuyến tính.
print("==========Cau 9==========")
diabetes_data = load_diabetes()
X = diabetes_data.data
y = diabetes_data.target
# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()
# Huấn luyện mô hình với dữ liệu huấn luyện
model.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 10/ Đầu tiên, bạn sẽ cần chia tập dữ liệu thành một tập huấn luyện (bao gồm 422 bệnh nhân đầu tiên)
# và một tập kiểm tra (20 bệnh nhân cuối cùng).
print("==========Cau 10==========")
X = diabetes_data.data  # Dữ liệu đầu vào
y = diabetes_data.target  # Nhãn mục tiêu
# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train = X[:422]  # 422 bệnh nhân đầu tiên
y_train = y[:422]
X_test = X[422:442]  # 20 bệnh nhân cuối cùng
y_test = y[422:442]
# Kiểm tra kích thước của các tập dữ liệu
print("Kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)

# 11/ Bây giờ, hãy áp dụng tập huấn luyện để dự đoán mô hình?
print("==========Cau 11==========")
# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()
# Huấn luyện mô hình
model.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
predictions = model.predict(X_test)
# In ra kết quả dự đoán
print("Giá trị dự đoán:", predictions)

# 12/ Làm thế nào để lấy được mười hệ số b đã được tính toán sau khi mô hình được huấn luyện?
print("==========Cau 12===========")
# Lấy các hệ số b
coefficients = model.coef_  # Hệ số cho các biến độc lập
intercept = model.intercept_  # Hệ số chặn
# In ra các hệ số
print("Các hệ số b:", coefficients)
print("Hệ số chặn:", intercept)

# 13/ Nếu bạn áp dụng tập kiểm tra vào dự đoán hồi quy tuyến tính, bạn sẽ nhận được một loạt các giá trị mục tiêu để
# so sánh với các giá trị thực tế đã quan sát.
print("==========Cau 13===========")
# Tải dữ liệu bệnh tiểu đường
diabetes_data = load_diabetes()
X = diabetes_data.data  # Dữ liệu đầu vào
y = diabetes_data.target  # Nhãn mục tiêu
# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train = X[:422]  # 422 bệnh nhân đầu tiên
y_train = y[:422]
X_test = X[422:442]  # 20 bệnh nhân cuối cùng
y_test = y[422:442]
# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()
# Huấn luyện mô hình
model.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Kiểm tra kích thước của y_test và y_pred
print("Kích thước y_test:", y_test.shape)
print("Kích thước y_pred:", y_pred.shape)
# So sánh giá trị dự đoán với giá trị thực tế
print("Giá trị thực tế:", y_test)
print("Giá trị dự đoán:", y_pred)
# Tính toán các chỉ số đánh giá
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Sai số tuyệt đối trung bình (MAE): {mae}')
print(f'R-bình phương (R²): {r2}')

# 14: Kiểm tra độ chính xác của mô hình dự đoán bằng các chỉ số như RMSE và R², cũng như phân tích phần dư.
print("==========Cau 14==========")

# Tải dữ liệu bệnh tiểu đường
diabetes_data = load_diabetes()
X = diabetes_data.data  # Dữ liệu đầu vào
y = diabetes_data.target  # Nhãn mục tiêu
# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()
# Huấn luyện mô hình
model.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Tính toán RMSE và R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'Sai số bình phương trung bình (RMSE): {rmse}')
print(f'R-bình phương (R²): {r2}')

# 15: Bắt đầu với mô hình hồi quy tuyến tính chỉ sử dụng một yếu tố sinh lý (ví dụ: tuổi).
print("==========Cau 15==========")
# Chọn yếu tố sinh lý là tuổi (age)
X_age = X[:, 0].reshape(-1, 1)  # Giả sử tuổi là biến đầu tiên trong tập dữ liệu
# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_age, y, test_size=0.2, random_state=42)
# Khởi tạo mô hình hồi quy tuyến tính
model_age = LinearRegression()
# Huấn luyện mô hình
model_age.fit(X_train_age, y_train_age)
# Dự đoán trên tập kiểm tra
y_pred_age = model_age.predict(X_test_age)
# Tính toán RMSE và R² cho mô hình tuổi
rmse_age = np.sqrt(mean_squared_error(y_test_age, y_pred_age))
r2_age = r2_score(y_test_age, y_pred_age)
print(f'Sai số bình phương trung bình (RMSE) cho tuổi: {rmse_age}')
print(f'R-bình phương (R²) cho tuổi: {r2_age}')

# 16: Tạo 10 mô hình hồi quy tuyến tính cho từng yếu tố sinh lý trong tập dữ liệu bệnh tiểu đường và trực quan
# hóa kết quả để có cái nhìn tổng quát hơn.
print("==========Cau 16==========")
# Danh sách các yếu tố sinh lý
physiological_factors = ['age', 'bmi', 'bp']
# Khởi tạo một từ điển để lưu kết quả
results = {}
# Tạo mô hình hồi quy tuyến tính cho mỗi yếu tố
for i, factor in enumerate(physiological_factors):
    X_factor = X[:, i].reshape(-1, 1)  # Chọn yếu tố sinh lý
    X_train_factor, X_test_factor, y_train_factor, y_test_factor = train_test_split(X_factor, y, test_size=0.2,
                                                                                    random_state=42)
    model_factor = LinearRegression()
    model_factor.fit(X_train_factor, y_train_factor)
    y_pred_factor = model_factor.predict(X_test_factor)
    # Tính toán RMSE và R²
    rmse_factor = np.sqrt(mean_squared_error(y_test_factor, y_pred_factor))
    r2_factor = r2_score(y_test_factor, y_pred_factor)
    results[factor] = {'RMSE': rmse_factor, 'R²': r2_factor}
    # Vẽ biểu đồ
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test_factor, y_test_factor, color='blue', label='Giá trị thực tế')
    plt.plot(X_test_factor, y_pred_factor, color='red', label='Giá trị dự đoán')
    plt.title(f'Hồi quy tuyến tính cho {factor}')
    plt.xlabel(factor)
    plt.ylabel('Chỉ số bệnh tiểu đường')
    plt.legend()
    plt.show()
# Hiển thị kết quả
print(results)

#17: Tải tập dữ liệu ung thư vú từ thư viện scikit-learn và in ra các khóa của từ điển.
print("==========Cau 17==========")
# Tải tập dữ liệu ung thư vú
cancer_data = load_breast_cancer()
# In ra các khóa của từ điển
print(cancer_data.keys())

#18 Kiểm tra kích thước của dữ liệu và đếm số lượng khối u “benign” và “malignant”.
print("==========Cau 18==========")
# Kiểm tra kích thước của dữ liệu
print("Kích thước của dữ liệu:", cancer_data.data.shape)
# Đếm số lượng khối u benign và malignant
labels, counts = np.unique(cancer_data.target, return_counts=True)
benign_count = counts[0]  # Khối u benign
malignant_count = counts[1]  # Khối u malignant
print(f"Số lượng khối u benign: {benign_count}")
print(f"Số lượng khối u malignant: {malignant_count}")

#19: Chia dữ liệu thành tập huấn luyện và tập kiểm tra, sau đó đánh giá hiệu suất của mô hình với số lượng hàng
# xóm khác nhau (từ 1 đến 10) và trực quan hóa kết quả.
print("==========Cau 19===========")
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=42)
# Danh sách để lưu trữ độ chính xác
accuracy = []
# Đánh giá hiệu suất với số lượng hàng xóm từ 1 đến 10
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), accuracy, marker='o')
plt.title('Hiệu suất của KNN với số lượng hàng xóm khác nhau')
plt.xlabel('Số lượng hàng xóm (k)')
plt.ylabel('Độ chính xác')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

#20: Tải thư viện mglearn, sử dụng tập dữ liệu make_forge và so sánh hồi quy
print("==========Cau 20===========")
# Tạo tập dữ liệu make_forge
X, y = make_forge()
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Khởi tạo mô hình hồi quy logistic
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# Khởi tạo mô hình Linear SVC
svc = LinearSVC()
svc.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred_log_reg = log_reg.predict(X_test)
y_pred_svc = svc.predict(X_test)
# Tính toán độ chính xác
accuracy_log_reg = np.mean(y_pred_log_reg == y_test)
accuracy_svc = np.mean(y_pred_svc == y_test)
print(f'Độ chính xác của hồi quy logistic: {accuracy_log_reg:.2f}')
print(f'Độ chính xác của Linear SVC: {accuracy_svc:.2f}')
# Vẽ biểu đồ để so sánh
plt.figure(figsize=(8, 5))
# Vẽ quyết định của hồi quy logistic
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title('Hồi quy Logistic')
xlim = plt.xlim()
ylim = plt.ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                     np.linspace(ylim[0], ylim[1], 100))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.xlim(xlim)
plt.ylim(ylim)
# Vẽ quyết định của Linear SVC
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title('Linear SVC')
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.xlim(xlim)
plt.ylim(ylim)
plt.tight_layout()
plt.show()


# 21/ Chúng ta sẽ áp dụng SVM cho nhận diện hình ảnh. Tập huấn luyện của chúng ta sẽ là một nhóm hình ảnh được gán
# nhãn của khuôn mặt con người. Bây giờ hãy bắt đầu bằng cách nhập và in ra mô tả của nó.
print("==========Cau 21===========")
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Giảm chiều dữ liệu bằng PCA
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Khởi tạo và huấn luyện mô hình SVC
clf = SVC(kernel='rbf', class_weight='balanced')
clf.fit(X_train_pca, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test_pca)

# In ra báo cáo phân loại
print(classification_report(y_test, y_pred, target_names=target_names))


# 22/ Nhìn vào nội dung của đối tượng lfw_people, chúng ta có các thuộc tính sau: images, data và target.
print("==========Cau 22==========")
# Kiểm tra các thuộc tính của đối tượng lfw_people
print("Số lượng hình ảnh:", lfw_people.images.shape[0])
print("Kích thước mỗi hình ảnh:", lfw_people.images.shape[1:])
print("Dữ liệu hình ảnh (dạng vector):", lfw_people.data.shape)
print("Số lượng nhãn:", lfw_people.target.shape[0])


# 23/ Trước khi học, hãy vẽ một số khuôn mặt. Vui lòng định nghĩa một hàm trợ giúp
print("==========Cau 23==========")
def plot_faces(images, titles=None, n_row=2, n_col=5):
    """Hàm để vẽ hình ảnh khuôn mặt."""
    plt.figure(figsize=(n_col * 2, n_row * 2))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i] if titles is not None else "")
        plt.xticks(())
        plt.yticks(())
    plt.show()

# Vẽ một số hình ảnh khuôn mặt
plot_faces(lfw_people.images, titles=lfw_people.target)

# 24/ Việc triển khai SVC có nhiều tham số quan trọng; có lẽ tham số liên quan nhất là kernel. Để bắt đầu, chúng ta sẽ sử dụng kernel đơn giản nhất, đó là kernel tuyến tính.
print("==========Cau 24=========")
# Khởi tạo mô hình SVC với kernel tuyến tính
model = SVC(kernel='linear')
# In ra thông tin mô hình
print(model)

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

# 25/ Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
print("===========Cau 25==========")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 26/ Định nghĩa hàm để đánh giá K-fold cross-validation
print("===========Cau 26============")
def evaluate_k_fold(X, y, k=5):
    kf = KFold(n_splits=k)
    results = []

    for train_index, test_index in kf.split(X):
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        # Huấn luyện mô hình
        clf = SVC(kernel='rbf', class_weight='balanced')
        clf.fit(X_train_k, y_train_k)

        # Dự đoán và đánh giá
        y_pred_k = clf.predict(X_test_k)
        report = classification_report(y_test_k, y_pred_k, target_names=target_names, output_dict=True)
        results.append(report['accuracy'])

    return np.mean(results)

# 27/ Định nghĩa hàm để huấn luyện và đánh giá
print("==========Cau 27==========")
def train_and_evaluate(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = clf.predict(X_test)

    # In ra báo cáo phân loại
    print(classification_report(y_test, y_pred, target_names=target_names))



# 28/ Huấn luyện và đánh giá
print("==========Cau 28===========")
# Gọi hàm K-fold cross-validation
k_fold_accuracy = evaluate_k_fold(X, y, k=5)
print("K-fold Cross-Validation Accuracy:", k_fold_accuracy)
# Gọi hàm huấn luyện và đánh giá
train_and_evaluate(X_train, y_train, X_test, y_test)

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names


# 29: Định nghĩa hàm để tạo ra mảng nhãn mới cho các khuôn mặt có kính và không có kính.
print("==========Cau 29===========")
def create_glasses_target(y, glasses_labels):
    """Tạo mảng nhãn mới cho các khuôn mặt có kính."""
    return np.array([1 if label in glasses_labels else 0 for label in y])

# 30: Chia lại tập dữ liệu và huấn luyện một bộ phân loại SVC mới với mảng nhãn mới.
print("==========Cau 30===========")
# Giả sử glasses_labels là danh sách các nhãn cho các khuôn mặt có kính
glasses_labels = [0, 1]  # Thay thế bằng các nhãn thực tế cho khuôn mặt có kính (ví dụ: 0 cho không có kính, 1 cho có kính)

# Tạo mảng nhãn mới
new_target = create_glasses_target(y, glasses_labels)

# Kiểm tra các lớp duy nhất trong new_target
unique_classes = np.unique(new_target)
print("Unique classes in new_target:", unique_classes)
print("Number of unique classes:", len(unique_classes))

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, new_target, test_size=0.25, random_state=42)

# Tạo bộ phân loại SVC mới
svc_2 = SVC(kernel='linear')
svc_2.fit(X_train, y_train)

# 31: Kiểm tra hiệu suất của mô hình bằng cách sử dụng cross-validation và báo cáo độ chính xác trung bình.
print("==========Cau 31===========")
# Hàm để đánh giá cross-validation
def evaluate_cross_validation(model, X, y, k=5):
    scores = cross_val_score(model, X, y, cv=k)
    print(f"Mean accuracy with {k}-fold cross-validation: {scores.mean():.3f}")

# Kiểm tra hiệu suất
evaluate_cross_validation(svc_2, X_train, y_train, 5)


import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

# 32: Tách riêng các hình ảnh của cùng một người
print("==========Cau 32==========")
person_indices = np.unique(y)  # Lấy các chỉ số của các người khác nhau
X_train = []
y_train = []
X_test = []
y_test = []

for person in person_indices:
    person_images = X[y == person]  # Lấy tất cả hình ảnh của người này
    if person_images.shape[0] > 10:  # Đảm bảo có đủ hình ảnh
        # Tách 10 hình ảnh từ chỉ số 30 đến 39 cho tập kiểm tra
        X_test.extend(person_images[30:40])  # Lấy hình ảnh từ chỉ số 30 đến 39
        y_test.extend([person] * 10)  # Nhãn cho các hình ảnh này
        # Sử dụng các hình ảnh còn lại cho tập huấn luyện
        X_train.extend(np.delete(person_images, np.arange(30, 40), axis=0))
        y_train.extend([person] * (person_images.shape[0] - 10))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Huấn luyện mô hình SVC
svc_2 = SVC(kernel='linear')
svc_2.fit(X_train, y_train)

# Hiển thị kết quả
print("==========Kết quả==========")
# Đánh giá mô hình trên tập huấn luyện
train_accuracy = svc_2.score(X_train, y_train)
print(f"Accuracy on training set: {train_accuracy:.2f}")

# Đánh giá mô hình trên tập kiểm tra
test_accuracy = svc_2.score(X_test, y_test)
print(f"Accuracy on test set: {test_accuracy:.2f}")

# Dự đoán trên tập kiểm tra
y_pred = svc_2.predict(X_test)

# In báo cáo phân loại
print("Classification report:")
print(metrics.classification_report(y_test, y_pred, target_names=target_names))

# In ma trận nhầm lẫn
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

# Tách dữ liệu (giả sử bạn đã thực hiện bước này trước đó)
# X_train, y_train, X_test, y_test đã được định nghĩa và huấn luyện mô hình svc_2

# 33: Kiểm tra hình ảnh bị phân loại sai
print("===========Cau 33===========")

# Dự đoán trên tập kiểm tra
y_pred = svc_2.predict(X_test)  # Sử dụng svc_2

# Kiểm tra hình ảnh bị phân loại sai
errors = np.where(y_pred != y_test)[0]  # Lấy chỉ số của các hình ảnh bị phân loại sai
print("Indices of incorrectly classified images:", errors)

# Kiểm tra kích thước của X_test
print("Shape of X_test:", X_test.shape)

# Định hình lại dữ liệu từ mảng thành ma trận 64 x 64 nếu có đủ hình ảnh
if X_test.shape[0] > 0 and X_test.shape[1] == 4096:  # Kiểm tra nếu có hình ảnh trong X_test
    X_test_reshaped = X_test.reshape(-1, 64, 64)  # Định hình lại thành 64x64
else:
    print("Error: X_test must have shape (n_samples, 4096) to reshape to (n_samples, 64, 64).")
    X_test_reshaped = None  # Đặt thành None nếu không thể định hình lại

# Hàm để vẽ hình ảnh
def print_faces(images, titles=None, n_row=2, n_col=5):
    """Hàm để vẽ hình ảnh khuôn mặt."""
    plt.figure(figsize=(n_col * 2, n_row * 2))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i] if titles is not None else "")
        plt.xticks(())
        plt.yticks(())
    plt.show()

# Vẽ các hình ảnh bị phân loại sai
if len(errors) > 0 and X_test_reshaped is not None:  # Kiểm tra nếu có hình ảnh bị phân loại sai
    eval_faces = [X_test_reshaped[i] for i in errors]  # Định hình lại các hình ảnh bị sai
    print_faces(eval_faces, titles=y_pred[errors])  # Vẽ các hình ảnh bị phân loại sai
else:
    print("No errors in classification or unable to reshape images.")

# Định nghĩa hàm train_and_evaluate
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on test set:")
    print(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

# Gọi hàm train_and_evaluate với mô hình và dữ liệu
# train_and_evaluate(svc_2, X_train, X_test, y_train, y_test)  # Uncomment để chạy
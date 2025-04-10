import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score

# 1. Tải dữ liệu chữ số viết tay từ sklearn.datasets
print("----------Cau 1------------")
digits = load_digits()
x = digits.data # Ma trận dữ liệu vơi s64 thuộc tính (8x8pixel)
y = digits.target #V ector mục tiêu chứa nhãn chữ số (0-9)
print("Các khóa của tập dữ liệu: ", digits.keys())
print("Kích thước dữ liệu x: ", x.shape)
print("Kích thước nhãn y: ", y.shape)

# 2. Định nghĩa hàm để viết các chữ số
print("----------Cau 2------------")
def plot_digits(data, lables, num_rows = 2, num_clos= 5, title = "Chữ số viết tay"):
    fig, axes = plt.subplots(num_rows, num_clos, figsize = (10, 4))
    for i, ax in enumerate(axes.ravel()):
        if i < len(data):
            ax.imshow(data[i].reshape(8, 8), cmap='gray')
            ax.set_title(f'Nhãn: {lables[i]}')
            ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

print("\nHiển thị 10 chữ số đầu tiên: ")
plot_digits(x, y, num_rows=2, num_clos=5)

# 3. Hàm hiển thị hình ảnh chữ số
print("----------Cau 3------------")
def plot_pca_scatter(x, y, title="PCA của tập dữ liệu chữ số viết tay"):
    #Khởi tạo mô hình PCA và giảm dữ liệu xuống 2 thành phần chính
    pca = PCA(n_components=2)
    # giảm dữ liệu xuống 2 thành phần chính
    x_pca = pca.fit_transform(x)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c =y, cmap='tab10', alpha=0.6, edgecolors='w')
    plt.colorbar(scatter, label='Lớp chữ số/Cụm')
    plt.xlabel('Thành phần chính 1')
    plt.ylabel('Thành phần chính 2')
    plt.title(title)
    plt.grid(True)
    plt.show()

    return x_pca, pca
# 4. Thực tập PCA và vẽ kết quả
print("----------Cau 4------------")
print("\n Vẽ biểu đồ phân tán 2D sau khi áp dụng PCA: ")
x_pca, pca_model = plot_pca_scatter(x, y)


# 5. Vẽ các thành phần PCA dưới dạng hình ảnh
print("----------Cau 5------------")

def plot_pca_components():
    pca = PCA(n_components=10)  # Lấy 10 thành phần chính
    pca.fit(x)
    components = pca.components_ #Xem kích thước

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(components[i].reshape(8, 8), cmap='gray')  # Hiển thị từng thành phần PCA
        ax.set_title(f'Thành phần {i + 1}')
        ax.axis('off')  # Ẩn trục tọa độ

    plt.suptitle("10 thành phần chính dưới dạng hình ảnh 8x8:")
    plt.tight_layout()  # Căn chỉnh bố cục
    plt.show()  # Hiển thị tất cả hình cùng lúc

print("\nHiển thị 10 hình ảnh PCA đầu tiên dưới dạng hình ảnh 8x8: ")
plot_pca_components()


# 6. Thực hiện phân cụm từ k-means
print("----------Cau 6------------")


# 7. Tách tập dữ liệu thành tập huấn luyện và kiểm tra
print("----------Cau 7------------")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # Chia dữ liệu thành 80% huấn luyện và 20% kiểm tra
print("Kích thước tập huấn luyện:", X_train.shape)  # In ra kích thước của tập huấn luyện
print("Kích thước tập kiểm tra:", X_test.shape)  # In ra kích thước của tập kiểm tra

# 8. Thử nghiệm với tham số n_init của k-means
print("----------Cau 8------------")
def experiment_with_n_init(X_train, y_train, n_clusters = 10):
    n_init_values = [1, 5, 10, 20]  #danh sách các giá trị thử nghiệm
    ari_score =[]

    print("\n Thử nghiệm với các giá trị n_init khác nhau cho k-means: ")
    for n_init in n_init_values: #lặp qua từng giá trị
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        kmeans.fit(X_train)
        clusters_labels = kmeans.labels_ #Lấy nhãn cụm của các điểm dữ liệu sau khi phân cụm
        ari = adjusted_rand_score(y_train, clusters_labels)
        ari_score.append(ari) # Lưu chỉ số ARI vào danh sách
        print(f"n_init = {n_init}, Chỉ số adjusted rand index: {ari:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(n_init_values, ari_score, marker='o', linestyle='-', color='b')
    plt.xlabel("Giá trị n_init")
    plt.ylabel("Chỉ số adjusted rand index")
    plt.title("Ảnh hưởng của n_init đến hiệu suất phân cụm k-means")
    plt.grid(True)
    plt.show()

experiment_with_n_init(X_train, y_train)
# 9. In nhãn cụm của dữ liệu huấn luyện
print("----------Cau 9------------")
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42) #khởi tạo mô hình k-means
kmeans.fit(X_train) #Huấn luyện mô hình K-Means
train_cluster_labels = kmeans.labels_ # Lấy nhãn cụm của từng mẫu dữ liệu trong tập huấn luyện

print("\n Nhãn cụm của dữ liệu huấn luyện (20 mẫu đầu tiên): ")
print(train_cluster_labels[:20])

print("\n Hiển thị một số chữ số huấn luyện với nhãn cụm dự đoán: ")
plot_digits(X_train[:10], train_cluster_labels[:10], title="Chữ số huấn luyện với nhãn cụm dự đoán")

# 10. Dự đoán cụm cho dữ liệu huấn luyện
print("----------Cau 10------------")
predicted_train_labels = kmeans.predict(X_train)  # Dự đoán cụm của tập huấn luyện
print("\n Nhãn cụm dự đoán cho dữ liệu huấn luyện (20 mẫu đầu tiên, dùng predict):")
print(predicted_train_labels[:20])

# 11. Định nghĩa hàm hiển thị hình ảnh từ mỗi cụm
print("----------Cau 11------------")
def print_cluster(X, cluster_lables, n_clusters=10, images_per_cluster=10):
    for cluster in range(n_clusters): # Duyệt qua từng cụm từ 0 đến n_clusters - 1
        cluster_indices = np.where(cluster_lables == cluster)[0]
        if len(cluster_indices) > 0:  # Kiểm tra nếu cụm có ít nhất một phần tử
            selected_indices = cluster_indices[:min(images_per_cluster, len(cluster_indices))] # Chọn tối đa images_per_cluster mẫu từ cụm hiện tại
            selected_images = X[selected_indices]  # Lấy hình ảnh tương ứng với chỉ số đã chọn
            selected_lables = cluster_lables[selected_indices]  # Lấy nhãn cụm tương ứng
            # Xác định số hàng và số cột để hiển thị hình ảnh
            num_rows = 2
            num_cols = 5
            print(f"\n Cụm  {cluster} (hiển thị {len(selected_indices)} hình ảnh): ")
            plot_digits(selected_images, selected_lables, num_rows, num_cols, title=f"Chữ số trong cụm {cluster}")
        else:
            print(f"\n Cụm {cluster} trống." )

print("\n Hiển thị 10 hình ảnh từ mỗi cụm (dữ liệu huấn luyện):")
print_cluster(X_train, train_cluster_labels)


# 12. Đánh giá hiệu suất bằng Adjusted Rand Index
print("----------Cau 12------------")
# Dự đoán nhãn cụm cho tập kiểm tra bằng mô hình K-Means đã huấn luyện
test_cluster_lables = kmeans.predict(X_test)
# Tính chỉ số Adjusted Rand Index (ARI) cho tập huấn luyện
# ARI đo lường mức độ tương đồng giữa các cụm được mô hình tạo ra và nhãn thực tế
train_ari = adjusted_rand_score(y_train, train_cluster_labels)
# Tính chỉ số Adjusted Rand Index (ARI) cho tập kiểm tra
# So sánh nhãn thực tế (y_test) với nhãn cụm dự đoán (test_cluster_labels)
test_ari = adjusted_rand_score(y_test, test_cluster_lables)

print(f"Chỉ số adjusted rand index cho tập huấn luyện: {train_ari:.4f}")
print(f"Chỉ số adjusted rand index cho tập kiểm tra: {test_ari:.4f}")

print("\nGiải thích")
print("Chỉ số adjusted rand index đo lường độ tương đồng giữa các cụm dự đoán và nhãn thực tế.")
print("Nó có giá trị từ -1 đến 1, trong đó 1 là phân cụm hoàn hảo, 0 là ngẫu nhiên, giá trị âm là tệ hơn ngẫu nhiên.")
print("Vì chúng ta có nhãn thực tế (lớp chữ số từ 0 - 9) chúng ta có thể tính ARI để đánh giá phân cụm k-means.")
print("Điều chỉnh theo yếu tố ngẫu nhiên, nên nó phản ánh chính xác chất lượng phân cụm")

#13 Vẽ lưới để hiển thị vùng phân cụm trong không gian 2D
print("----------Cau 13------------")
def plot_kmeans_decision_boundary(X, y, kmeans, title="Vùng phân cụm k-means trong không gian 2D"):
    #Giảm chiều dữ liệu về 2 D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    #Tạo lưới các điểm trong không gian 2D
    h = 0.5 #Bước của lưới
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    xx, yy, = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    #Chuyển đổi các điểm lưới về không gian gốc 64 chiều để dự đoán các cụm
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original = pca.inverse_transform(mesh_points)
    mesh_labels = kmeans.predict(mesh_points_original)

    #Vẽ vùng phân cụm
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, mesh_labels.reshape(xx.shape), cmap='tab10', alpha=0.3)

    #Vẽ các điểm dữ liệu
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', edgecolors='k', alpha=0.6)
    plt.colorbar(scatter, label='Nhãn cụm')
    plt.xlabel('Thành phần chính 1')
    plt.ylabel('Thành phần chính 2')
    plt.title(title)
    plt.show()

print("\n Vẽ vùng phân cụm k-means trong không gian 2D: ")
plot_kmeans_decision_boundary(X_train, train_cluster_labels, kmeans, title="Vùng phân cụm k-means trn dữ liệu huấn luyện")
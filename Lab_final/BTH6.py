#Bài 1: Định nghĩa MDP với 3 trạng thái, 2 hành động

# Các trạng thái
states = [ "S0", "S1", "S2"]
# Các hành động
actions = ["a0", "a1"]

# Mô tả xác suất chuyển và phần thưởng
transitions_probs = {
    "S0": {
        "a0": {"S0": 0.5, "S1": 0.0, "S2": 0.5},
        "a1": {"S0": 0.0, "S1": 0.0, "S2": 1.0}
    },
    "S1": {
        "a0": {"S0": 0.7, "S1": 0.1, "S2": 0.2},
        "a1": {"S0": 0.0, "S1": 0.95, "S2": 0.05}
    },
    "S2": {
        "a0": {"S0": 0.4, "S1": 0.0, "S2": 0.6},
        "a1": {"S0": 0.3, "S1": 0.3, "S2": 0.4}
    }
}

# Hàm phần thưởng R(s, a, s')
def reward(s, a, s_prime):
    if a == "a0":
        return 0
    else:  #a == "a1"
        return -1


# In ra MDP để kiểm tra
print("=====BÀI 1=====\n")
print("Tập trạng thái:", states)
print("Tập hành động:", actions)
print("\nHàm chuyển tiếp T(s, a, s'):")
for s in states:
    for a in actions:
        print(f"Từ {s}, hành động {a}:")
        for s_prime in states:
            prob = transitions_probs[s][a][s_prime]
            print(f"  -> {s_prime}: {prob}")

print("\nHàm phần thưởng R(s, a, s'):")
for s in states:
    for a in actions:
        for s_prime in states:
            r = reward(s, a, s_prime)
            print(f"R({s}, {a}, {s_prime}) = {r}")




#Bài 2: sử dụng value interation để tìm chính sách tối ưu
print("\n=====BÀI 2=====\n")
# Các trạng thái
states = [ "S0", "S1", "S2"]
# Các hành động
actions = ["a0", "a1"]

# Mô tả xác suất chuyển và phần thưởng
transitions_probs = {
    "S0": {
        "a0": {"S0": 0.5, "S1": 0.0, "S2": 0.5},
        "a1": {"S0": 0.0, "S1": 0.0, "S2": 1.0}
    },
    "S1": {
        "a0": {"S0": 0.7, "S1": 0.1, "S2": 0.2},
        "a1": {"S0": 0.0, "S1": 0.95, "S2": 0.05}
    },
    "S2": {
        "a0": {"S0": 0.4, "S1": 0.0, "S2": 0.6},
        "a1": {"S0": 0.3, "S1": 0.3, "S2": 0.4}
    }
}

# Hàm phần thưởng R(s, a, s')
def reward(s, a, s_prime):
    if a == "a0":
        return 0
    else:  #a == "a1"
        return -1

#Tham số cho value iteration
GAMMA = 0.9 # hệ số chiết khấu
NUM_ITERATIONS = 10  # số bước lập

#Value iteration
def value_iteration():
    #khởi tạo V(s) = 0
    V = {s: 0 for s in states}
    policy = {s: None for s in states}

    for iteration in range(NUM_ITERATIONS):
        V_new = V.copy()
        for s in states:
            max_value = float("-inf")
            best_action = None
            for a in actions:
                value = 0
                for s_prime in states:
                    prob = transitions_probs[s][a][s_prime]
                    value += prob * (reward(s, a, s_prime) + GAMMA * V[s_prime])
                if value > max_value:
                    max_value = value
                    best_action = a
            V_new[s] = max_value
            policy[s] = best_action
        V = V_new

        #in gia trị V(s) sau mỗi bước lập
        print(f"\n Lặp {iteration + 1}:")
        for s in states:
            print(f"V({s}) = {V[s]: .4f}")

    return V, policy

#chạy thuật toán
V, policy = value_iteration()

#in kết quả chính sách tối ưu
print("\n Chính sách tối ưu:")
for s in states:
    print(f"π({s}) = {policy[s]}")



#Bài 3 mô hình "robot cleaning" và tìm chính sách tối ưu
print("\n=====BÀI 3=====\n")
# Các trạng thái
states = [ "S0", "S1", "S2"]
# Các hành động
actions = ["di_chuyen", "don_dep"]

# Mô tả xác suất chuyển và phần thưởng
transitions_probs = {
    "S0": {
        "di_chuyen": {"S0": 0.5, "S1": 0.0, "S2": 0.5},
        "don_dep": {"S0": 0.0, "S1": 0.0, "S2": 1.0}
    },
    "S1": {
        "di_chuyen": {"S0": 0.7, "S1": 0.1, "S2": 0.2},
        "don_dep": {"S0": 0.0, "S1": 0.95, "S2": 0.05}
    },
    "S2": {
        "di_chuyen": {"S0": 0.4, "S1": 0.0, "S2": 0.6},
        "don_dep": {"S0": 0.3, "S1": 0.3, "S2": 0.4}
    }
}

# Hàm phần thưởng R(s, a, s')
def reward(s, a, s_prime):
    if a == "di_chuyen":
        return 0
    else:  #a == "don_dep"
        if s == "S1":
            return 2  #S1 là nơi bẩn nhất dọn dẹp nhận thưởng cao
        else:
            return -1  #S0, S2 không cần dọn mất công sức

#Tham số cho value iteration
GAMMA = 0.9 # hệ số chiết khấu
NUM_ITERATIONS = 10  # số bước lập

#Value iteration
def value_iteration():
    #khởi tạo V(s) = 0
    V = {s: 0 for s in states}
    policy = {s: None for s in states}

    for iteration in range(NUM_ITERATIONS):
        V_new = V.copy()
        for s in states:
            max_value = float("-inf")
            best_action = None
            for a in actions:
                value = 0
                for s_prime in states:
                    prob = transitions_probs[s][a][s_prime]
                    value += prob * (reward(s, a, s_prime) + GAMMA * V[s_prime])
                if value > max_value:
                    max_value = value
                    best_action = a
            V_new[s] = max_value
            policy[s] = best_action
        V = V_new

        #in gia trị V(s) sau mỗi bước lập
        print(f"\n Lặp {iteration + 1}:")
        for s in states:
            print(f"V({s}) = {V[s]: .4f}")

    return V, policy

#chạy thuật toán
V, policy = value_iteration()

#in kết quả chính sách tối ưu
print("\n Chính sách tối ưu:")
for s in states:
    print(f"π({s}) = {policy[s]}")
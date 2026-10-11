import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes_ = None

    def _impurity(self, y):
        m = len(y)
        if m == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        probabilities = counts / m

        if self.criterion == 'gini':
            return 1.0 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            p_pos = probabilities[probabilities > 0]
            return -np.sum(p_pos * np.log2(p_pos))
        else:
            raise ValueError(f"Неизвестный критерий: {self.criterion}")

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None

        current_impurity = self._impurity(y)
        best_gain = -1.0
        best_feat, best_thresh = None, None

        for feat_idx in range(n):
            values = X[:, feat_idx]
            thresholds = np.unique(values)
            if len(thresholds) <= 1:
                continue

            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                imp_l = self._impurity(y[left_mask])
                imp_r = self._impurity(y[right_mask])

                child_impurity = (n_l / m) * imp_l + (n_r / m) * imp_r
                gain = current_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = np.argmax(np.bincount(y, minlength=self.n_classes_))
            return Node(value=leaf_value)

        feat, thresh = self._best_split(X, y)
        if feat is None:
            leaf_value = np.argmax(np.bincount(y, minlength=self.n_classes_))
            return Node(value=leaf_value)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def print_tree(self, node=None, depth=0, feature_names=None):
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}└── Класс: {node.value}")
        else:
            feat_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
            print(f"{indent}├── Если {feat_name} <= {node.threshold:.3f}:")
            self.print_tree(node.left, depth + 1, feature_names)
            print(f"{indent}└── Иначе ({feat_name} > {node.threshold:.3f}):")
            self.print_tree(node.right, depth + 1, feature_names)


def train_test_split_np(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    n_test = int(n * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def evaluate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1


def main():
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ МЕТОДА РЕШАЮЩИХ ДЕРЕВЬЕВ")
    print("Команда 01: Ярик, Даша | Машинное обучение")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 300

    X0 = np.random.randn(n_samples // 2, 2) * 1.1 + np.array([-1.2, -0.8])
    X1 = np.random.randn(n_samples // 2, 2) * 1.0 + np.array([1.5, 1.2])
    X0_extra = np.random.randn(n_samples // 4, 2) * 0.7 + np.array([1.8, -1.8])
    X0 = np.vstack([X0, X0_extra])

    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))

    feature_names = ["Доход (норм.)", "Кредитный рейтинг (норм.)"]

    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.3, random_state=42)

    tree = DecisionTreeClassifier(max_depth=4, min_samples_split=5, criterion='gini')
    tree.fit(X_train, y_train)

    print("\nСтруктура обученного дерева решений:")
    tree.print_tree(feature_names=feature_names)

    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)

    acc_tr, prec_tr, rec_tr, f1_tr = evaluate_metrics(y_train, y_pred_train)
    acc_te, prec_te, rec_te, f1_te = evaluate_metrics(y_test, y_pred_test)

    print("\n" + "=" * 50)
    print(f"Accuracy (Train): {acc_tr:.4f} | Accuracy (Test): {acc_te:.4f}")
    print(f"Precision: {prec_te:.4f} | Recall: {rec_te:.4f} | F1: {f1_te:.4f}")
    print("=" * 50)

    depths = list(range(1, 12))
    train_scores = []
    test_scores = []

    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, min_samples_split=2, criterion='gini')
        m.fit(X_train, y_train)
        train_scores.append(evaluate_metrics(y_train, m.predict(X_train))[0])
        test_scores.append(evaluate_metrics(y_test, m.predict(X_test))[0])

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = tree.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                color='red', label='Класс 0 (Train)', alpha=0.6, edgecolors='k', s=35)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                color='blue', label='Класс 1 (Train)', alpha=0.6, edgecolors='k', s=35)
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                color='darkred', marker='^', label='Класс 0 (Test)', alpha=0.8, s=45)
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                color='darkblue', marker='^', label='Класс 1 (Test)', alpha=0.8, s=45)

    plt.title(f"Разделяющие границы (max_depth=4, Test Acc={acc_te:.2%})", fontsize=12)
    plt.xlabel(feature_names[0], fontsize=11)
    plt.ylabel(feature_names[1], fontsize=11)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(depths, train_scores, 'o-', color='navy', label='Обучающая выборка (Train)', linewidth=2)
    plt.plot(depths, test_scores, 's--', color='crimson', label='Тестовая выборка (Test)', linewidth=2)
    plt.axvline(x=4, color='green', linestyle=':', label='Оптимальная глубина (max_depth=4)', linewidth=2)
    plt.title("Зависимость точности от максимальной глубины (Переобучение)", fontsize=12)
    plt.xlabel("Максимальная глубина дерева (max_depth)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.xticks(depths)
    plt.ylim([0.65, 1.02])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = "01-Метод решающих деревьев/src/decision_tree_output.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

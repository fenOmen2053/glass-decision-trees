

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def load_glass2():
    """Load glass2 with pmlb, with a reproducible fallback for environments without pmlb."""
    try:
        from pmlb import fetch_data
        data = fetch_data('glass2')
        source = "PMLB: fetch_data('glass2')"
        return data, source
    except Exception as exc:
        print("PMLB loading failed, using UCI fallback:", exc)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
        glass = pd.read_csv(url, header=None, names=columns)
        # glass2 is the binary window-glass task: float processed vs non-float processed.
        data = glass[glass['Type'].isin([1, 2, 3])].copy()
        data['target'] = data['Type'].isin([1, 3]).astype(int)
        data = data.drop(columns=['Id', 'Type'])
        source = "UCI fallback reconstructed as glass2"
        return data, source


def prepare_xy(data):
    """Split the dataset into X features and y target."""
    if 'target' not in data.columns:
        raise ValueError("The dataset must contain a target column.")
    X = data.drop(columns=['target'])
    y = data['target']
    if not np.issubdtype(y.dtype, np.number):
        y = pd.factorize(y)[0]
    y = pd.Series(y, name='target')
    return X, y


def evaluate_parameter_grid(X, y):
    """Evaluate random_state, criterion and max_depth combinations."""
    rows = []
    models = {}
    for random_state in [0, 42, 7]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=random_state, stratify=y
        )
        for criterion in ['gini', 'entropy']:
            for max_depth in [2, 3, 5, None]:
                clf = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    random_state=random_state,
                )
                clf.fit(X_train, y_train)
                train_pred = clf.predict(X_train)
                test_pred = clf.predict(X_test)
                rows.append({
                    'random_state': random_state,
                    'criterion': criterion,
                    'max_depth': 'None' if max_depth is None else max_depth,
                    'train_accuracy': accuracy_score(y_train, train_pred),
                    'test_accuracy': accuracy_score(y_test, test_pred),
                    'train_f1_weighted': f1_score(y_train, train_pred, average='weighted'),
                    'test_f1_weighted': f1_score(y_test, test_pred, average='weighted'),
                    'tree_depth': clf.get_depth(),
                    'n_leaves': clf.get_n_leaves(),
                })
                models[(random_state, criterion, max_depth)] = (clf, X_train, X_test, y_train, y_test)
    results = pd.DataFrame(rows).sort_values(
        ['test_accuracy', 'test_f1_weighted', 'train_accuracy'],
        ascending=[False, False, True],
    )
    return results, models


def plot_feature_space_partition(X, y, feature_names, output_path):
    """Visualize decision regions using the two most important features."""
    X2 = X[feature_names]
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=0.30, random_state=0, stratify=y
    )
    clf2 = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)
    clf2.fit(X_train, y_train)

    x_min, x_max = X2[feature_names[0]].min() - 0.2, X2[feature_names[0]].max() + 0.2
    y_min, y_max = X2[feature_names[1]].min() - 0.2, X2[feature_names[1]].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350))
    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(7, 5.5))
    plt.contourf(xx, yy, Z, alpha=0.35)
    plt.scatter(
        X_train[feature_names[0]], X_train[feature_names[1]],
        c=y_train, marker='o', label='train', edgecolor='k', linewidths=0.3,
    )
    plt.scatter(
        X_test[feature_names[0]], X_test[feature_names[1]],
        c=y_test, marker='^', label='test', edgecolor='k', linewidths=0.3,
    )
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f'Feature space partition using {feature_names[0]} and {feature_names[1]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def explain_one_prediction(model, X_test, y_test, feature_names, class_names):
    """Return readable decision-path rules for one test sample."""
    sample = X_test.iloc[[0]]
    sample_index = sample.index[0]
    true_class = int(y_test.loc[sample_index])
    pred_class = int(model.predict(sample)[0])
    proba = model.predict_proba(sample)[0]

    node_indicator = model.decision_path(sample)
    leaf_id = model.apply(sample)[0]
    tree_feature = model.tree_.feature
    threshold = model.tree_.threshold

    path_rules = []
    for node_id in node_indicator.indices:
        if node_id == leaf_id:
            continue
        feature_id = tree_feature[node_id]
        feature_name = feature_names[feature_id]
        value = sample.iloc[0, feature_id]
        operator = '<=' if value <= threshold[node_id] else '>'
        path_rules.append(f"{feature_name} = {value:.3f} {operator} {threshold[node_id]:.3f}")

    return {
        'sample_index': sample_index,
        'features': sample.iloc[0].to_dict(),
        'true_class': class_names[true_class],
        'predicted_class': class_names[pred_class],
        'predicted_probabilities': {
            class_names[i]: round(float(p), 3) for i, p in enumerate(proba)
        },
        'path_rules': path_rules,
    }



def create_dtreeviz_visualization(model, X_train, y_train, X_test, feature_names, class_names, output_path='dtreeviz_visualization.png'):
    """Create a dtreeviz visualization. If dtreeviz is unavailable, create a local
    dtreeviz-style figure with split histograms, threshold markers, leaf purity,
    and a sample prediction path.
    """
    output_path = Path(output_path)
    sample = X_test.iloc[[0]]
    sample_index = sample.index[0]

    # Try the real dtreeviz package first. This works in Google Colab after:
    # %pip install dtreeviz graphviz
    try:
        from dtreeviz import model as dtreeviz_model
        viz_model = dtreeviz_model(
            model,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            target_name='target',
            class_names=class_names,
        )
        view = viz_model.view(scale=0.8, orientation='TD')
        svg_path = output_path.with_suffix('.svg')
        view.save(str(svg_path))
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_path), write_to=str(output_path), dpi=180)
        except Exception:
            print(f"dtreeviz SVG saved to {svg_path}; PNG conversion skipped.")
        return
    except Exception as exc:
        print("dtreeviz is not available in this environment, creating a fallback visualization:", exc)

    # Fallback visualization with the same information as dtreeviz: distributions,
    # thresholds, leaf composition, and the local prediction path.
    from matplotlib.gridspec import GridSpec
    tree_ = model.tree_
    root_feature_id = int(tree_.feature[0])
    root_feature = feature_names[root_feature_id]
    root_threshold = float(tree_.threshold[0])
    root_value = float(sample.iloc[0, root_feature_id])
    pred = int(model.predict(sample)[0])
    proba = model.predict_proba(sample)[0]
    node_indicator = model.decision_path(sample)
    leaf_id = int(model.apply(sample)[0])

    path_rules = []
    for node_id in node_indicator.indices:
        if tree_.feature[node_id] < 0:
            continue
        fid = int(tree_.feature[node_id])
        fname = feature_names[fid]
        thresh = float(tree_.threshold[node_id])
        val = float(sample.iloc[0, fid])
        op = '<=' if val <= thresh else '>'
        path_rules.append(f'{fname} = {val:.3f} {op} {thresh:.3f}')

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, height_ratios=[1.45, 1.1, 0.9], hspace=0.42, wspace=0.25)

    ax_tree = fig.add_subplot(gs[0, :])
    plot_tree(model, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=7, max_depth=3, ax=ax_tree)
    ax_tree.set_title('dtreeviz-style tree overview: split logic and class purity (first 4 levels)', fontsize=14, pad=12)

    ax_hist = fig.add_subplot(gs[1, 0])
    bins = np.linspace(X_train[root_feature].min(), X_train[root_feature].max(), 16)
    ax_hist.hist(X_train.loc[y_train == 0, root_feature], bins=bins, alpha=0.65, label=class_names[0])
    ax_hist.hist(X_train.loc[y_train == 1, root_feature], bins=bins, alpha=0.65, label=class_names[1])
    ax_hist.axvline(root_threshold, linestyle='--', linewidth=2, label=f'threshold = {root_threshold:.3f}')
    ax_hist.scatter([root_value], [ax_hist.get_ylim()[1] * 0.92], marker='v', s=120, label=f'sample #{sample_index}')
    ax_hist.set_title(f'Root split distribution: {root_feature} <= {root_threshold:.3f}')
    ax_hist.set_xlabel(root_feature)
    ax_hist.set_ylabel('Samples')
    ax_hist.legend(fontsize=8)

    ax_leaf = fig.add_subplot(gs[1, 1])
    leaf_ids = model.apply(X_train)
    unique_leaves = sorted(np.unique(leaf_ids).tolist())
    count0, count1, labels = [], [], []
    for lid in unique_leaves:
        idx = leaf_ids == lid
        labels.append(str(lid))
        count0.append(int(np.sum(y_train[idx] == 0)))
        count1.append(int(np.sum(y_train[idx] == 1)))
    x = np.arange(len(unique_leaves))
    ax_leaf.bar(x, count0, label=class_names[0])
    ax_leaf.bar(x, count1, bottom=count0, label=class_names[1])
    ax_leaf.set_xticks(x)
    ax_leaf.set_xticklabels(labels, rotation=90, fontsize=7)
    ax_leaf.set_xlabel('Leaf node id')
    ax_leaf.set_ylabel('Training samples')
    ax_leaf.set_title('Leaf purity and final class composition')
    ax_leaf.legend(fontsize=8)

    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')
    text = [
        f'Prediction path for sample #{sample_index}', '',
        *[f'• {line}' for line in path_rules], '',
        f'Leaf node: {leaf_id}',
        f'Predicted class: {class_names[pred]}',
        f'Probabilities: {class_names[0]}={proba[0]:.3f}, {class_names[1]}={proba[1]:.3f}',
    ]
    ax_text.text(0.02, 0.98, '\n'.join(text), va='top', fontsize=11, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5', edgecolor='#999999'))
    ax_text.set_title('Local explainability from the decision path')

    fig.suptitle('dtreeviz visualization for glass2 Decision Tree', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

def main():
    data, source = load_glass2()
    X, y = prepare_xy(data)
    feature_names = list(X.columns)
    class_names = ['non_float_processed', 'float_processed'] if set(np.unique(y)) == {0, 1} else [str(c) for c in np.unique(y)]

    print("Dataset source:", source)
    print("Dataset shape:", data.shape)
    print("Class distribution:")
    print(pd.Series(y).value_counts().sort_index())

    results, models = evaluate_parameter_grid(X, y)
    print("\nParameter comparison:")
    print(results.to_string(index=False))

    # Best model selected from the comparison: high test accuracy with controlled depth.
    best_key = (0, 'gini', 5)
    best_model, X_train, X_test, y_train, y_test = models[best_key]
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print("\nBest model:", best_key)
    print("Train accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nTest classification report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    importance = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance)

    print("\nText rules from the tree:")
    print(export_text(best_model, feature_names=feature_names, decimals=3))

    explanation = explain_one_prediction(best_model, X_test, y_test, feature_names, class_names)
    print("\nExplainability example:")
    print(explanation)

    # Save figures.
    results.to_csv('parameter_results.csv', index=False)

    pd.Series(y).value_counts().sort_index().plot(kind='bar', figsize=(6, 4), title='glass2 class distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred), display_labels=class_names).plot(
        ax=ax, values_format='d', colorbar=False
    )
    plt.title('Confusion matrix - best Decision Tree')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=180)
    plt.close()

    plt.figure(figsize=(18, 10))
    plot_tree(best_model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8)
    plt.title('Decision Tree visualization')
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=180)
    plt.close()

    importance.sort_values().plot(kind='barh', figsize=(7, 4.5), title='Feature importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=180)
    plt.close()

    top2 = importance.index[:2].tolist()
    plot_feature_space_partition(X, y, top2, 'feature_space_partition.png')

    create_dtreeviz_visualization(
        best_model, X_train, y_train, X_test, feature_names, class_names,
        output_path='dtreeviz_visualization.png'
    )


if __name__ == '__main__':
    main()

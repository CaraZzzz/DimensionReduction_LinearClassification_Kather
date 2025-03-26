import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import mahalanobis

def apply_LDA(X_train, y_train, X_test, n_dimension):
    """
    Apply LDA for dimension reduction
    
    Args:
        X_train: training data
        y_train: training labels
        X_test: test data
        n_dimension: number of dimensions to reduce to
        
    Returns:
        X_train_LDA: transformed training data
        X_test_LDA: transformed test data
    """
    # Calculate the maximum possible number of components
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    max_components = min(n_features, n_classes - 1)
    
    # Adjust n_dimension if it exceeds the maximum
    n_dimension = min(n_dimension, max_components)
    
    # Initialize LDA
    lda = LDA(n_components=n_dimension)
    
    # Fit LDA on training data
    X_train_LDA = lda.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_LDA = lda.transform(X_test)
    
    return X_train_LDA, X_test_LDA

def apply_pca(X, n_components):
    """
    Apply PCA for dimension reduction
    
    Args:
        X: input data
        n_components: number of components to keep
        
    Returns:
        X_pca: transformed data
        pca: fitted PCA object for transforming new data
    """
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print explained variance ratio
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Total explained variance with {n_components} components: {explained_var:.2f}%")
    
    return X_pca, pca, scaler

def apply_pca_LDA(X_train, y_train, X_test, n_pca, n_lda):
    X_train_pca, pca, scaler = apply_pca(X_train, n_pca)
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    X_train_LDA, X_test_LDA = apply_LDA(X_train_pca, y_train, X_test_pca, n_lda)
    return X_train_LDA, X_test_LDA

def train_knn_classifier(k, X_train, y_train, X_test, y_test):
    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model
    print(f"Training KNN classifier with k={k}...")
    knn.fit(X_train, y_train)
    
    # Evaluate on training set
    train_predictions = knn.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"Training set accuracy: {train_accuracy:.4f}")
    
    # Evaluate on test set
    test_predictions = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test set accuracy: {test_accuracy:.4f}")
    
    return knn, train_accuracy, test_accuracy

def mahalanobis_classifier(X_train, y_train, X_test, y_test):
    # Calculate the mean and covariance matrix for each class
    classes = np.unique(y_train)
    class_means = {}
    class_cov_inv = {}
    
    for cls in classes:
        X_cls = X_train[y_train == cls]
        class_means[cls] = np.mean(X_cls, axis=0)
        class_cov_inv[cls] = np.linalg.inv(np.cov(X_cls, rowvar=False))
    
    def predict(X):
        predictions = []
        for x in X:
            min_dist = float('inf')
            best_class = None
            for cls in classes:
                mean = class_means[cls]
                cov_inv = class_cov_inv[cls]
                dist = mahalanobis(x, mean, cov_inv)
                if dist < min_dist:
                    min_dist = dist
                    best_class = cls
            predictions.append(best_class)
        return np.array(predictions)
    
    # Predict and calculate accuracy for training set
    y_train_pred = predict(X_train)
    train_accuracy = np.mean(y_train_pred == y_train)
    
    # Predict and calculate accuracy for test set
    y_test_pred = predict(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)
    
    return train_accuracy, test_accuracy

def softmax_classifier(X_train, y_train, X_test, y_test):
    # Initialize the logistic regression model with a multinomial option for softmax
    softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    
    # Train the model
    softmax.fit(X_train, y_train)
    
    # Predict and calculate accuracy for training set
    y_train_pred = softmax.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Predict and calculate accuracy for test set
    y_test_pred = softmax.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    return train_accuracy, test_accuracy


def Draw_component_accuracy(train_acc, test_acc, train_acc_noDR, test_acc_noDR,
                            best_k=None, best_n_pca=None, best_n_lda=None,
                            x_start=1, 
                            classification=None, Dimension_reduction=None):
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    n_up = len(train_acc)
    x_values = range(x_start, n_up)
    # Plot accuracies with PCA
    plt.plot(x_values, train_acc[x_start:], 'r-', marker='o', label='Training Accuracy')
    plt.plot(x_values, test_acc[x_start:], 'b-', marker='o', label='Test Accuracy')
    
    # Add horizontal lines for non-PCA accuracies
    plt.axhline(y=train_acc_noDR, color='r', linestyle='--', label='Training Accuracy (No Dimention Reduction)')
    plt.axhline(y=test_acc_noDR, color='b', linestyle='--', label='Test Accuracy (No Dimention Reduction)')
    
    # Add annotation for the best point
    best_test_acc = np.max(test_acc)
    best_n = np.argmax(test_acc)
    plt.annotate(f'Best: n={best_n}\nacc={best_test_acc:.4f}',
                xy=(best_n, best_test_acc),
                xytext=(10, 10),
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Number of Dimention Reduction Components')
    plt.ylabel('Accuracy')

    plt.xticks(x_values)
    plt.legend()
    plt.grid(True)


    if Dimension_reduction == 'pca':
        title_dimension_reduction = f'PCA'
        save_dimension_reduction = f'PCA'
    if Dimension_reduction == 'lda':
        title_dimension_reduction = f'LDA'
        save_dimension_reduction = f'LDA'
    if Dimension_reduction == 'pca_lda':
        title_dimension_reduction = f'PCA(N_pca={best_n_pca})+LDA'
        save_dimension_reduction = f'PCALDA_Npca{best_n_pca}'


    if classification == 'knn':
        title_classification = f'KNN(k={best_k})'
        save_classification = f'KNN_k{best_k}'
    if classification == 'mahalanobis':
        title_classification = f'Mahalanobis'
        save_classification = f'Maha'
    if classification == 'softmax':
        title_classification = f'Softmax'
        save_classification = f'Softmax'

    plt.title(f'{title_dimension_reduction}+{title_classification} ')
    plt.savefig(f'{save_dimension_reduction}_{save_classification}.png')

    plt.show()

def PCA_knn(k_up, n_up, X_train, y_train, X_test, y_test):
    # Initialize matrices to store accuracies
    train_acc_matrix = np.zeros((k_up, n_up))
    test_acc_matrix = np.zeros((k_up, n_up))
    
    # Store best parameters
    best_k = 1
    best_n = 1
    best_test_acc = 0
    
    # Try different combinations of k and n_components
    for k in range(1, k_up):
        for n_components in range(1, n_up):
            # Apply PCA
            X_train_pca, pca, scaler = apply_pca(X_train, n_components)
            X_test_scaled = scaler.transform(X_test)
            X_test_pca = pca.transform(X_test_scaled)
            
            # Train and evaluate model
            _, train_acc, test_acc = train_knn_classifier(k, X_train_pca, y_train, X_test_pca, y_test)
            
            # Store accuracies
            train_acc_matrix[k, n_components] = train_acc
            test_acc_matrix[k, n_components] = test_acc
            
            # Update best parameters if necessary
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_k = k
                best_n = n_components

    # Get accuracies without PCA for the best k
    _, train_acc_noDR, test_acc_noDR = train_knn_classifier(best_k, X_train, y_train, X_test, y_test)
    
    # Get accuracies for best k with different n_components
    train_acc_best_k = train_acc_matrix[best_k, :]
    test_acc_best_k = test_acc_matrix[best_k, :]
    
    Draw_component_accuracy(train_acc_best_k, test_acc_best_k, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=best_k, best_n_pca=None, best_n_lda=None,
                            x_start=1,
                            classification="knn", Dimension_reduction="pca")
    return best_k, best_n, best_test_acc

def PCA_mahalanobis(n_pca, X_train, y_train, X_test, y_test):
    train_acc_noDR, test_acc_noDR = mahalanobis_classifier(X_train, y_train, X_test, y_test)
    print(f"\nBaseline (No PCA) accuracies:")
    print(f"Training accuracy: {train_acc_noDR:.4f}")
    print(f"Test accuracy: {test_acc_noDR:.4f}\n")

    train_accuracy_list = [0] * n_pca
    test_accuracy_list = [0] * n_pca
    for n_components in range(2, n_pca):
        # Fit PCA on training data and transform all datasets
        X_train_pca, pca, scaler = apply_pca(X_train, n_components)
        
        # Transform test set using the same PCA and scaler
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)
        
        train_acc, test_acc = mahalanobis_classifier(X_train_pca, y_train, X_test_pca, y_test)
        train_accuracy_list[n_components] = train_acc
        test_accuracy_list[n_components] = test_acc

        print(f"Accuracies with {n_components} PCA components:")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}\n")
    
    train_accuracy_array = np.array(train_accuracy_list)
    test_accuracy_array = np.array(test_accuracy_list)
    Draw_component_accuracy(train_accuracy_array, test_accuracy_array, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=None, best_n_pca=None, best_n_lda=None,
                            x_start=2,
                            classification="mahalanobis", Dimension_reduction="pca")
    
    max_test_accuracy = max(test_accuracy_list[1:])
    best_n_components = test_accuracy_list[1:].index(max_test_accuracy) + 1
    return best_n_components, max_test_accuracy

def PCA_softmax(n_pca, X_train, y_train, X_test, y_test):
    train_acc_noDR, test_acc_noDR = softmax_classifier(X_train, y_train, X_test, y_test)
    print(f"\nBaseline (No PCA) accuracies:")
    print(f"Training accuracy: {train_acc_noDR:.4f}")
    print(f"Test accuracy: {test_acc_noDR:.4f}\n")

    train_accuracy_list = [0] * n_pca
    test_accuracy_list = [0] * n_pca
    for n_components in range(1, n_pca):
        # Fit PCA on training data and transform all datasets
        X_train_pca, pca, scaler = apply_pca(X_train, n_components)
        
        # Transform test set using the same PCA and scaler
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)
        
        train_acc, test_acc = softmax_classifier(X_train_pca, y_train, X_test_pca, y_test)
        train_accuracy_list[n_components] = train_acc
        test_accuracy_list[n_components] = test_acc

        print(f"Accuracies with {n_components} PCA components:")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}\n")

    # Find the maximum test accuracy and its corresponding n_components
    max_test_accuracy = max(test_accuracy_list[1:])
    best_n_components = test_accuracy_list[1:].index(max_test_accuracy) + 1

    train_accuracy_array = np.array(train_accuracy_list)
    test_accuracy_array = np.array(test_accuracy_list)
    Draw_component_accuracy(train_accuracy_array, test_accuracy_array, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=None, best_n_pca=None, best_n_lda=None,
                            x_start=1,
                            classification="softmax", Dimension_reduction="pca")
    return best_n_components, max_test_accuracy

def LDA_knn(k_up, n_up, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    max_components = min(n_features, n_classes - 1)
    n_up = min(n_up, max_components+1)

    # Initialize matrices to store accuracies
    train_acc_matrix = np.zeros((k_up, n_up))
    test_acc_matrix = np.zeros((k_up, n_up))
    
    # Store best parameters
    best_k = 1
    best_n = 1
    best_test_acc = 0
    
    # Try different combinations of k and n_components
    for k in range(1, k_up):
        for n_components in range(1, n_up):
            X_train_LDA, X_test_LDA = apply_LDA(X_train, y_train, X_test, n_components)
            
            # Train and evaluate model
            _, train_acc, test_acc = train_knn_classifier(k, X_train_LDA, y_train, X_test_LDA, y_test)
            
            # Store accuracies
            train_acc_matrix[k, n_components] = train_acc
            test_acc_matrix[k, n_components] = test_acc
            
            # Update best parameters if necessary
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_k = k
                best_n = n_components

    # Get accuracies without LDA for the best k
    _, train_acc_noDR, test_acc_noDR = train_knn_classifier(best_k, X_train, y_train, X_test, y_test)
    
    # Get accuracies for best k with different n_components
    train_acc_best_k = train_acc_matrix[best_k, :]
    test_acc_best_k = test_acc_matrix[best_k, :]
    
    Draw_component_accuracy(train_acc_best_k, test_acc_best_k, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=best_k, best_n_pca=None, best_n_lda=None,
                            x_start=1,
                            classification="knn", Dimension_reduction="lda")
    return best_k, best_n, best_test_acc

def LDA_mahalanobis(n_up, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    max_components = min(n_features, n_classes - 1)
    n_up = min(n_up, max_components+1)

    train_acc_noDR, test_acc_noDR = mahalanobis_classifier(X_train, y_train, X_test, y_test)
    print(f"\nBaseline (No PCA) accuracies:")
    print(f"Training accuracy: {train_acc_noDR:.4f}")
    print(f"Test accuracy: {test_acc_noDR:.4f}\n")

    train_accuracy_list = [0] * n_up
    test_accuracy_list = [0] * n_up
    for n_components in range(2, n_up):
        X_train_LDA, X_test_LDA = apply_LDA(X_train, y_train, X_test, n_components)
        
        train_acc, test_acc = mahalanobis_classifier(X_train_LDA, y_train, X_test_LDA, y_test)
        train_accuracy_list[n_components] = train_acc
        test_accuracy_list[n_components] = test_acc

        print(f"Accuracies with {n_components} LDA components:")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}\n")
    
    train_accuracy_array = np.array(train_accuracy_list)
    test_accuracy_array = np.array(test_accuracy_list)
    Draw_component_accuracy(train_accuracy_array, test_accuracy_array, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=None, best_n_pca=None, best_n_lda=None,
                            x_start=2,
                            classification="mahalanobis", Dimension_reduction="lda")
    
    max_test_accuracy = max(test_accuracy_list[1:])
    best_n_components = test_accuracy_list[1:].index(max_test_accuracy) + 1
    return best_n_components, max_test_accuracy

def LDA_softmax(n_up, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    max_components = min(n_features, n_classes - 1)
    n_up = min(n_up, max_components+1)

    train_acc_noDR, test_acc_noDR = softmax_classifier(X_train, y_train, X_test, y_test)
    print(f"\nBaseline (No PCA) accuracies:")
    print(f"Training accuracy: {train_acc_noDR:.4f}")
    print(f"Test accuracy: {test_acc_noDR:.4f}\n")

    train_accuracy_list = [0] * n_up
    test_accuracy_list = [0] * n_up
    for n_components in range(1, n_up):
        # Fit PCA on training data and transform all datasets
        X_train_LDA, X_test_LDA = apply_LDA(X_train, y_train, X_test, n_components)
        
        train_acc, test_acc = softmax_classifier(X_train_LDA, y_train, X_test_LDA, y_test)
        train_accuracy_list[n_components] = train_acc
        test_accuracy_list[n_components] = test_acc

        print(f"Accuracies with {n_components} LDA components:")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}\n")

    # Find the maximum test accuracy and its corresponding n_components
    max_test_accuracy = max(test_accuracy_list[1:])
    best_n_components = test_accuracy_list[1:].index(max_test_accuracy) + 1

    train_accuracy_array = np.array(train_accuracy_list)
    test_accuracy_array = np.array(test_accuracy_list)
    Draw_component_accuracy(train_accuracy_array, test_accuracy_array, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=None, best_n_pca=None, best_n_lda=None,
                            x_start=1,
                            classification="softmax", Dimension_reduction="lda")
    return best_n_components, max_test_accuracy

def PCA_LDA_knn(k_up, n_up_pca, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    max_components = min(n_features, n_classes - 1)
    n_up_lda = max_components+1
    n_up_pca = max(n_up_lda, n_up_pca)

    # Initialize matrices to store accuracies
    train_acc_matrix = np.zeros((k_up, n_up_pca, n_up_lda))
    test_acc_matrix = np.zeros((k_up, n_up_pca, n_up_lda))
    
    # Store best parameters
    best_k = 1
    best_n_pca = 1
    best_n_lda = 1
    best_test_acc = 0
    
    # Try different combinations of k and n_components
    for k in range(1, k_up):
        for n_pca in range(1, n_up_pca):
            X_train_pca, pca, scaler = apply_pca(X_train, n_pca)
            X_test_scaled = scaler.transform(X_test)
            X_test_pca = pca.transform(X_test_scaled)

            for n_lda in range(1, n_up_lda):
                # Fit LDA on training data and transform all datasets
                X_train_LDA, X_test_LDA = apply_LDA(X_train_pca, y_train, X_test_pca, n_lda)
                
                _, train_acc, test_acc = train_knn_classifier(k, X_train_LDA, y_train, X_test_LDA, y_test)
                
                # Store accuracies
                train_acc_matrix[k, n_pca, n_lda] = train_acc
                test_acc_matrix[k, n_pca, n_lda] = test_acc
            
            # Update best parameters if necessary
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_k = k
                best_n_pca = n_pca
                best_n_lda = n_lda

    # Get accuracies without LDA for the best k
    _, train_acc_noDR, test_acc_noDR = train_knn_classifier(best_k, X_train, y_train, X_test, y_test)
    
    # Get accuracies for best k with different n_components
    train_acc_best_k = train_acc_matrix[best_k, best_n_pca, :]
    test_acc_best_k = test_acc_matrix[best_k, best_n_pca, :]

    Draw_component_accuracy(train_acc_best_k, test_acc_best_k, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=best_k, best_n_pca=best_n_pca, best_n_lda=None,
                            x_start=1,
                            classification="knn", Dimension_reduction="pca_lda")
    return best_k, best_n_pca, best_test_acc

def PCA_LDA_mahalanobis(n_up_pca, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    max_components = min(n_features, n_classes - 1)
    n_up_lda = max_components+1
    n_up_pca = max(n_up_lda, n_up_pca)

    # Initialize matrices to store accuracies
    train_acc_matrix = np.zeros((n_up_pca, n_up_lda))
    test_acc_matrix = np.zeros((n_up_pca, n_up_lda))
    
    # Store best parameters
    best_n_pca = 1
    best_n_lda = 1
    best_test_acc = 0
    
    # Try different combinations of k and n_components
    for n_pca in range(n_up_lda, n_up_pca):
        X_train_pca, pca, scaler = apply_pca(X_train, n_pca)
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)

        for n_lda in range(2, n_up_lda):
            # Fit LDA on training data and transform all datasets
            X_train_LDA, X_test_LDA = apply_LDA(X_train_pca, y_train, X_test_pca, n_lda)
            
            train_acc, test_acc = mahalanobis_classifier(X_train_LDA, y_train, X_test_LDA, y_test)
            train_acc_matrix[n_pca, n_lda] = train_acc
            test_acc_matrix[n_pca, n_lda] = test_acc

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_n_pca = n_pca
                best_n_lda = n_lda
    
    train_acc_noDR, test_acc_noDR = mahalanobis_classifier(X_train, y_train, X_test, y_test)
    
    # Get accuracies for best k with different n_components
    train_acc_best_k = train_acc_matrix[best_n_pca, :]
    test_acc_best_k = test_acc_matrix[best_n_pca, :]

    Draw_component_accuracy(train_acc_best_k, test_acc_best_k, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=None, best_n_pca=best_n_pca, best_n_lda=None,
                            x_start=2,
                            classification="mahalanobis", Dimension_reduction="pca_lda")
    return best_n_pca, best_test_acc


def PCA_LDA_softmax(n_up_pca, X_train, y_train, X_test, y_test):
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    max_components = min(n_features, n_classes - 1)
    n_up_lda = max_components+1
    n_up_pca = max(n_up_lda, n_up_pca)

    # Initialize matrices to store accuracies
    train_acc_matrix = np.zeros((n_up_pca, n_up_lda))
    test_acc_matrix = np.zeros((n_up_pca, n_up_lda))
    
    # Store best parameters
    best_n_pca = 1
    best_n_lda = 1
    best_test_acc = 0
    
    # Try different combinations of k and n_components
    for n_pca in range(n_up_lda, n_up_pca):
        X_train_pca, pca, scaler = apply_pca(X_train, n_pca)
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)

        for n_lda in range(1, n_up_lda):
            # Fit LDA on training data and transform all datasets
            X_train_LDA, X_test_LDA = apply_LDA(X_train_pca, y_train, X_test_pca, n_lda)
            
            train_acc, test_acc = softmax_classifier(X_train_LDA, y_train, X_test_LDA, y_test)
            train_acc_matrix[n_pca, n_lda] = train_acc
            test_acc_matrix[n_pca, n_lda] = test_acc

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_n_pca = n_pca
                best_n_lda = n_lda  
    
    train_acc_noDR, test_acc_noDR = softmax_classifier(X_train, y_train, X_test, y_test)
    
    # Get accuracies for best k with different n_components
    train_acc_best_k = train_acc_matrix[best_n_pca, :]
    test_acc_best_k = test_acc_matrix[best_n_pca, :]

    Draw_component_accuracy(train_acc_best_k, test_acc_best_k, 
                            train_acc_noDR, test_acc_noDR, 
                            best_k=None, best_n_pca=best_n_pca, best_n_lda=None,
                            x_start=1,
                            classification="softmax", Dimension_reduction="pca_lda")
    return best_n_pca, best_test_acc

    

if __name__ == "__main__":
    # Read the CSV file
    # data = pd.read_csv("D:/MSc/Courses/EE6222/CA/archive/hmnist_28_28_RGB.csv")
    # data = pd.read_csv("D:/MSc/Courses/EE6222/CA/archive/hmnist_28_28_L.csv")
    # data = pd.read_csv("D:/MSc/Courses/EE6222/CA/archive/hmnist_64_64_L.csv")
    # data = pd.read_csv("D:/MSc/Courses/EE6222/CA/archive/hmnist_8_8_L.csv")
    data = pd.read_csv("D:/MSc/Courses/EE6222/CA/archive/hmnist_8_8_RGB.csv")
    
    # Print the shape of the data
    print("Data shape:", data.shape)

    # Separate features (X) and labels (y)
    X = data.drop('label', axis=1).to_numpy()
    y = data['label'].to_numpy()

    # Split data into 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the shapes to verify the split
    print("Data split sizes:")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(data)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(data)*100:.1f}%)\n")

    # PCA_knn
    # best_k, best_n, best_test_acc = PCA_knn(k_up=30, n_up=30, 
    #                                         X_train=X_train, y_train=y_train, 
    #                                         X_test=X_test, y_test=y_test)
    
    # PCA_mahalanobis
    # best_n_components, best_test_acc = PCA_mahalanobis(n_pca=20, 
    #                                                     X_train=X_train, y_train=y_train, 
    #                                                     X_test=X_test, y_test=y_test) 

    # PCA_softmax
    # best_n_components, best_test_acc = PCA_softmax(n_pca=120, 
    #                                                 X_train=X_train, y_train=y_train, 
    #                                                 X_test=X_test, y_test=y_test) 

    # LDA_knn
    # best_k, best_n, best_test_acc = LDA_knn(k_up=70, n_up=30, 
    #                                         X_train=X_train, y_train=y_train, 
    #                                         X_test=X_test, y_test=y_test)

    # LDA_mahalanobis
    # best_n_components, best_test_acc = LDA_mahalanobis(n_up=10, 
    #                                                     X_train=X_train, y_train=y_train, 
    #                                                     X_test=X_test, y_test=y_test)

    # LDA_softmax
    # best_n_components, best_test_acc = LDA_softmax(n_up=10, 
    #                                                 X_train=X_train, y_train=y_train, 
    #                                                 X_test=X_test, y_test=y_test)

    # PCA_LDA_knn
    best_k, best_n_pca,  best_test_acc = PCA_LDA_knn(k_up=30, n_up_pca=70, 
                                                    X_train=X_train, y_train=y_train, 
                                                    X_test=X_test, y_test=y_test)

    # PCA_LDA_mahalanobis
    # best_n_pca, best_test_acc = PCA_LDA_mahalanobis(n_up_pca=120, 
    #                                                 X_train=X_train, y_train=y_train, 
    #                                                 X_test=X_test, y_test=y_test)

    # PCA_LDA_softmax
    # best_n_pca, best_test_acc = PCA_LDA_softmax(n_up_pca=120, 
    #                                             X_train=X_train, y_train=y_train, 
    #                                             X_test=X_test, y_test=y_test)






# Standard libraries
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Scikit-learn libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

class get_data:
    def __init__(self):
        self.clinical = pd.read_csv("MGH_COVID_Clinical_Info.txt", delimiter = ";")
        self.olink = pd.read_csv("MGH_COVID_OLINK_NPX.txt", delimiter = ";")
        self.dataset = None

    def load_dataset(self):
        D0_data = self.olink[self.olink['Timepoint'] == 'D0']
        D0_proteins = D0_data.pivot_table(index = 'subject_id', columns = 'UniProt', values = 'NPX')
        D0_proteins.reset_index(inplace = True)
        D0_proteins = pd.merge(D0_proteins, self.clinical[['subject_id', 'COVID']], on = 'subject_id', how = 'left')
        
        assay_warn = D0_data[D0_data['Assay_Warning'] != 'PASS']
        assay_warn_proteins = assay_warn['UniProt'].unique()
        
        D0_proteins = D0_proteins.drop(columns = assay_warn_proteins, errors='ignore')
        self.dataset = D0_proteins.drop(columns = 'subject_id')
        
        return self.dataset
        
    def split_data(self, best_proteins):
        self.load_dataset
        self.dataset = self.dataset.loc[:, best_proteins + ['COVID']]

        X = self.dataset.drop('COVID', axis = 1)
        y = self.dataset['COVID']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 54)
        
        datasets = {
            'train': {
                'X': X_train, 
                'y': y_train
            },
            'test': {
                'X': X_test, 
                'y': y_test
            }
        }
        
        return datasets

class VolcanoPlot:
    def __init__(self, df):
        self.df = df
        self.ttest_df = None
        self.top_proteins = None
        self.significant_upregulated = None
        self.significant_downregulated = None

    def perform_ttest(self):
        disease_df = self.df[self.df['COVID'] == 1].drop('COVID', axis = 1)
        control_df = self.df[self.df['COVID'] == 0].drop('COVID', axis = 1)

        proteins = disease_df.columns

        fold_changes = []
        p_values = []

        for protein in proteins:
            disease_npx = disease_df[protein]
            control_npx = control_df[protein]

            fold_change = disease_npx.mean() - control_npx.mean()
            fold_changes.append(fold_change)

            t_stat, p_value = ttest_ind(disease_npx, control_npx)
            p_values.append(p_value)

        self.ttest_df = pd.DataFrame({
            'Proteins': proteins,
            'Fold Change': fold_changes,
            'P-value': p_values
        })

    def prepare_data(self):
        self.perform_ttest()

        self.ttest_df['-log10(P-value)'] = -np.log10(self.ttest_df['P-value'])
        self.ttest_df.sort_values(by='-log10(P-value)', ascending=False, inplace=True)

        self.top_proteins = self.ttest_df.head(100)
        self.significant_upregulated = self.ttest_df[(self.ttest_df['Fold Change'] > 1) & (self.ttest_df['-log10(P-value)'] > 1.3)]
        self.significant_downregulated = self.ttest_df[(self.ttest_df['Fold Change'] < -1) & (self.ttest_df['-log10(P-value)'] > 1.3)]

        return self.top_proteins['Proteins'], self.significant_upregulated['Proteins'], self.significant_downregulated['Proteins']

    def plot(self):
        plt.figure(figsize = (10, 8))
        plt.scatter(self.ttest_df['Fold Change'], self.ttest_df['-log10(P-value)'], label = 'Not Significant')

        low_p_value = self.ttest_df[self.ttest_df['-log10(P-value)'] > 1.3]
        plt.scatter(low_p_value['Fold Change'], low_p_value['-log10(P-value)'], color = 'lightblue', label = 'Low P-value')

        plt.scatter(self.significant_upregulated['Fold Change'], self.significant_upregulated['-log10(P-value)'], color = 'red', label = 'Significant Upregulated')
        plt.scatter(self.significant_downregulated['Fold Change'], self.significant_downregulated['-log10(P-value)'], color = 'green', label = 'Significant Downregulated')
        plt.scatter(self.top_proteins['Fold Change'], self.top_proteins['-log10(P-value)'], color = 'black', label = 'Top 100 Proteins')

        plt.title('Volcano Plot')
        plt.legend(loc = 'best')
        plt.xlabel('Fold Change')
        plt.ylabel('-log10(P-value)')

        plt.show()
    

    def plot_best_proteins(self, best_proteins):
        plt.figure(figsize = (10, 8))
        plt.scatter(self.ttest_df['Fold Change'], self.ttest_df['-log10(P-value)'], label = 'Other Proteins')
        
        best_proteins_df = self.ttest_df[self.ttest_df['Proteins'].isin(best_proteins)]
        plt.scatter(best_proteins_df['Fold Change'], best_proteins_df['-log10(P-value)'], color = 'red', label = f'Best {len(best_proteins)} Proteins')
    
        for i, protein in enumerate(best_proteins_df['Proteins']):
            plt.annotate(protein, (best_proteins_df['Fold Change'].iloc[i], best_proteins_df['-log10(P-value)'].iloc[i]))
       
        plt.title('Volcano Plot')
        plt.legend(loc = 'best')
        plt.xlabel('Fold Change')
        plt.ylabel('-log10(P-value)')
    
        plt.show()

class Logistic_Regression:
    def __init__(self, dataset_dic):
        self.df = dataset_dic
        self.X_train = self.df['train']['X']
        self.y_train = self.df['train']['y']
        self.X_test = self.df['test']['X']
        self.y_test = self.df['test']['y']
        
        self.lr = LogisticRegression(penalty = 'l1', solver = 'liblinear')
        self.coefficients = None
        self.selector = None
        
        self.X_train_best = None
        self.X_test_best = None
        self.best_param = None
        
    def train(self):
        self.lr.fit(self.X_train, self.y_train)
        self.coefficients = self.lr.coef_[0]
    
    def LASSO(self, len_proteins = None):
        if len_proteins != None:
            threshold = np.sort(np.abs(self.coefficients))[-len_proteins]
        else:
            threshold = None
        
        self.selector = SelectFromModel(estimator = self.lr, prefit = True, threshold = threshold)
        
        self.X_train_best = self.selector.transform(self.X_train)
        self.X_test_best = self.selector.transform(self.X_test)
        
        return self.X_train_best
        
    def plot_all_proteins(self):
        x = np.arange(len(self.coefficients))

        plt.bar(x, self.coefficients)
        plt.xlabel('Proteins')
        plt.ylabel('Weight')
        plt.title('Protein Weights')

        plt.show()
    
    def plot_best_proteins(self, num_proteins, best_proteins):
        selected_features = self.selector.get_support(indices=True)
        selected_coefficients = self.coefficients[selected_features]

        x = np.arange(len(selected_coefficients))

        plt.bar(x, selected_coefficients)
        plt.xticks(x, best_proteins, rotation = 45)
        plt.xlabel('Selected Proteins')
        plt.ylabel('Weight')
        plt.title(f'Weights of {num_proteins} Selected Proteins')

        plt.show()    
   
    def hyperparam_search(self, param_grid):
        lr = LogisticRegression()
        grid_search = GridSearchCV(lr, param_grid, cv = 10, scoring='accuracy')
        grid_search.fit(self.X_train_best, self.y_train)

        self.best_params = grid_search.best_params_

        print("Best Hyperparameters: ", self.best_params)
        
    def evaluate(self):
        best_model = LogisticRegression(C = self.best_params['C'], penalty = self.best_params['penalty'], solver = self.best_params['solver'])
        best_model.fit(self.X_train_best, self.y_train)
        y_pred = best_model.predict(self.X_test_best)

        self.lr.fit(self.X_train_best, self.y_train)
        y_pred = self.lr.predict(self.X_test_best)

        accuracy = accuracy_score(self.y_test, y_pred)

        y_pred_prob = best_model.predict_proba(self.X_test_best)[:, 1]
        auc = roc_auc_score(self.y_test, y_pred_prob)

        print("Test Accuracy:", accuracy)
        print("Test AUC:", auc)
        
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)

        return accuracy, auc, (fpr, tpr)
        
def get_protein_names(A, B):
    B = pd.DataFrame(B)
    
    assert A.shape[0] == B.shape[0], "Number of rows in A and B must be the same"
    assert A.shape[1] >= B.shape[1], "Number of columns in A must be greater than or equal to B"

    column_names = []
    for i in range(B.shape[1]):
        for col in A.columns:
            if np.all(A[col].to_numpy() == B.iloc[:, i].to_numpy()):
                column_names.append(col)
                break

    return column_names

def main():
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    got_data = get_data()
    df = got_data.load_dataset()
    
    volcano = VolcanoPlot(df)
    top_100, sig_up, sig_down = volcano.prepare_data()
    volcano.plot()
    
    best_proteins = set()
    best_proteins.update(top_100)
    best_proteins.update(sig_up)
    best_proteins.update(sig_down)
    best_proteins = list(best_proteins)
    
    dataset_dic = got_data.split_data(best_proteins)
    
    num_proteins = [1, 5, 8, 10, 12, 15, 25, 50]
    param_grid = {
        'C': np.logspace(-4, 4, 40),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    accuracy_values = []
    auc_values = []
    roc_curve_data = []
    
    best_accuracy = 0
    best_auc = 0
    best_num = 0
    
    lr = Logistic_Regression(dataset_dic)
    lr.train()
    lr.plot_all_proteins()
    
    for num in num_proteins:
        lr.train()
        lr.LASSO(num)
        lr.hyperparam_search(param_grid)
        accuracy, auc, fpr_tpr = lr.evaluate()
        
        accuracy_values.append(accuracy)
        auc_values.append(auc)
        roc_curve_data.append(fpr_tpr)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_auc = auc
            best_num = num
    
    lr.train()
    X = lr.LASSO(best_num)
    best_proteins = get_protein_names(dataset_dic['train']['X'], X)
        
    # Plot accuracy values
    plt.figure()
    plt.plot(num_proteins, accuracy_values, marker='o')
    plt.xlabel('Number of Proteins')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Different Numbers of Proteins')
    plt.ylim([min(accuracy_values) - 0.05, max(accuracy_values) + 0.05])
    plt.grid(True)
    plt.show()
    
    # Plot AUC values
    plt.figure()
    plt.plot(num_proteins, auc_values, marker='o')
    plt.xlabel('Number of Proteins')
    plt.ylabel('AUC')
    plt.title('AUC for Different Numbers of Proteins')
    plt.ylim([min(auc_values) - 0.05, max(auc_values) + 0.05])
    plt.grid(True)
    plt.show()
    
    # Plot ROC curve
    plt.figure()
    for i, (fpr, tpr) in enumerate(roc_curve_data):
        plt.plot(fpr, tpr, label = f'Num Proteins = {num_proteins[i]} (AUC = {auc_values[i]:.3f})')
    plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Different Numbers of Proteins')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot Weights of Best Proteins
    lr.plot_best_proteins(best_num, best_proteins)
    
    # Plot Volcano Plot with Best PRoteins
    # Plot Volcano w/ Best Proteins
    volcano.plot_best_proteins(best_proteins)
    print(f"The best {best_num} proteins: {best_proteins}")

if __name__ == "__main__":
    main()
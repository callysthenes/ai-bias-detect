import matplotlib.pyplot as plt
from random import randrange
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd 
import seaborn as sns  # Graphing
from scipy.stats import chi2_contingency
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from scipy.stats import chisquare
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix,roc_curve, auc
from pandas_profiling import ProfileReport 
from sklearn.metrics import PrecisionRecallDisplay


class Report():

    """ 
        NAME: Report
        ============

        DESCRIPTION: A report module based on Seaborn
        ==============================================================================
        Report takes the imported data set and returns a distribution of each of the 
        variables without having to call seaborn.pairplot.

        PARAMETERS: data (pandas dataframe) and dependent variable 
        ==============================================================================

        RETURNS: Pairplots, violin and heatmap graphs made in seaborn
        ==============================================================================

    """


    def profilereporting(self,data,dependent):
        """
        A detailed EDA made up in Seaborn to understand better the data composition
        Returns violin plots, pairplots, and heatmaps
        """
        report = ProfileReport(data, minimal=False)

        hist = data.hist(layout =(5,4), color = 'Pink', ec="black", figsize =(15,12), grid =False)
        plt.suptitle("Historgram plots for all numeric variables")
        plt.show()
        pass
        #Pairplot 
        pairplot = sns.pairplot(data);
        visual = pairplot.fig.suptitle("Pairplot", y=1.08);
        plt.show();
        pass

        #Heatmap 
        matrix = data.corr()
        mask = np.zeros_like(matrix)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots(figsize=(50,10))
        heatmap = sns.heatmap(matrix, center=0, fmt=".2f", square=True, annot=True, linewidth=.9, mask = mask,vmax=.8);
        plt.show()
        pass
        
        #Violinplot Categorial variables
        try:
            vars = data.select_dtypes(include = "object").columns
            plt.suptitle("Violin plots for all categorial")
            fig, axs = plt.subplots(1, len(vars), figsize=(35,4))
            for i in range(0,len(vars)):
                sns.violinplot( x=data[vars[i]],y=data[dependent], data=data,ax=axs[i],palette = 'rainbow_r')
                axs[i].set_ylabel(dependent)
                for tick in axs[i].get_xticklabels():
                    tick.set_rotation(90)
                    #for tick in axs[i].get_xticklabels():
                    #   tick.set_rotation(90)
            plt.show()
        except:
            print ("Violin plots are not supported with integer type, use a dataset with object type")
            pass

        return hist, visual, heatmap, report



class Backup():
    """ 
        NAME: Backup
        ============

        DESCRIPTION: A duplication method to backup the original dataset
        =========================================================================================

        PARAMETERS: data (pandas dataframe)
        =========================================================================================

        RETURNS: A new dataframe with backed up data
        =========================================================================================

    """
    def make_backup(self, data):
        backup_data = data.copy()


class EnhancedLabelEncoder(LabelEncoder):
    """ 
        NAME: LabelEncoder
        ============

        DESCRIPTION: Takes columns in original dataset and prompts user for a label in each of them
        ===========================================================================================

        PARAMETERS: data (pandas dataframe)
        ===========================================================================================

        RETURNS: Same dataframe with recoded names for columns
        ===========================================================================================

    """

    def fit_transform_columns(self, data):
        objList = data.select_dtypes(include = "object").columns
        for feat in objList:
            data[feat] = super().fit_transform(data[feat]) # it is a method from the parent class, to use the parent with super
        return data
    

class  Splitting():# has no state, otherwise there would be an __init__ method
    """ 
        NAME: Splitting
        ===============

        DESCRIPTION: Splits original dataset in X and y by removing all independent variables. X are the independent variables for a model, with the dependent already taken out
        y is the dependent variable already isolated
        ================================================================================================

        PARAMETERS: data (pandas dataframe), dependent (dependent variables)
        ================================================================================================

        RETURNS: X_train, X_test, y_train, y_test (with an 80-20% ratio for train and test respectively)
        ================================================================================================

    """
    def split_data(self,data,dependent):
        X = data.drop([dependent],axis=1)# axis: {0 or ‘index’, 1 or ‘columns’}, default 0
        y = data[dependent]
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test


class FairDetect():
    """ 
        NAME: FairDetect
        ================

        DESCRIPTION: A method framework to detect bias in classification datasets
        =========================================================================

        PARAMETERS: Check individual method
        =========================================================================

        RETURNS: Check individual method
        =========================================================================
    """

    def __init__(self, model,X_test,y_test):
        """Instatiation of model, X_test and y_test"""
        self.model = model
        self.X_test = X_test
        self.y_test = y_test


    def representation(self,sensitive, labels, predictions):
        """
        METHOD: representation
        ======================

        DESCRIPTION: Performs a chi-test and a xg-boost prediction to
        understand the relation between the original ratio, and the 
        predictions of the classifier. Based on this, the chi test is done
        between the attribute chosen (sensitive variable), and the target
        variable. It will automatically accept or reject the null hypothesis
        ===========================================================================

        PARAMETERS: The sensitive group, labels previously defined, and predictions
        ===========================================================================

        RETURNS: Accept or reject the null hypothesis between sensitive and target
        ===========================================================================
        """
        

        full_table = self.X_test.copy()
        sens_df = {}
        
        for i in labels:
            full_table['p'] = predictions
            full_table['t'] = self.y_test
            sens_df[labels[i]] = full_table[full_table[sensitive]==i]

        contigency_p = pd.crosstab(full_table[sensitive], full_table['t']) 
        cp, pp, dofp, expectedp = chi2_contingency(contigency_p) 
        contigency_pct_p = pd.crosstab(full_table[sensitive], full_table['t'], normalize='index')

        sens_rep = {}
        for i in labels:
            sens_rep[labels[i]] = (self.X_test[sensitive].value_counts()/self.X_test[sensitive].value_counts().sum())[i]
            
        labl_rep = {}
        for i in labels:
            labl_rep[str(i)] = (self.y_test.value_counts()/self.y_test.value_counts().sum())[i]

        
        fig = make_subplots(rows=1, cols=2)
        
        for i in labels:
            fig.add_trace(go.Bar(
            showlegend=False,
            x = [labels[i]],
            y= [sens_rep[labels[i]]]),row=1,col=1)
            
            fig.add_trace(go.Bar(
            showlegend=False,
            x = [str(i)],
            y= [labl_rep[str(i)]],
            marker_color=['purple','cyan'][i]),row=1,col=2)
        
        fig.update_layout(height=400, width=800) # nicer size of the output 

        c, p, dof, expected = chi2_contingency(contigency_p)
        cont_table = (tabulate(contigency_pct_p.T, headers=labels.values(), tablefmt='fancy_grid'))
        
        
        threshold = 0.5 #Classification report, a table with precision, recall, etc.
        y_predi = predictions >= threshold
        classification = classification_report(self.y_test, y_predi)
        
        #Improved Model
        
        print('Confusion Matrix')
        
        y_actual = pd.Series(self.y_test, name='Actual')
        y_predicted = pd.Series(y_predi, name='Predicted')
        confusion = pd.crosstab(np.array(self.y_test), np.array(y_predi), rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(confusion, annot=True,fmt=".1f")
        plt.show()
              
        
        # ROC, AUC
        metric = metrics.roc_auc_score(self.y_test, y_predi)
        
        plot_roc_curve(self.model, self.X_test,self.y_test)
        plt.plot([0,1], [0,1], color="navy", linestyle="--")
        plt.title("Receiver operating characteristic curve")
        plt.show()
        
        #PrecisionRecallCurve
        PrecisionRecallDisplay.from_estimator(self.model, self.X_test, self.y_test)
        plt.show()
        
        
        print(classification)
        print(f"AUC {metric}")
        
        return cont_table, sens_df, fig, p,classification, confusion, metric
            


    def ability(self, sens_df,labels):
        """
        METHOD: ability
        ======================

        DESCRIPTION: A method to obtain the TPR (True Positive Rate), TNR (True 
        negative Rate), FPR (False Positive Rate), FNR (True Negative Rate)
        and store it as a variable
        ===========================================================================

        PARAMETERS: sens_df returned in the representation method and the labels
        ===========================================================================

        RETURNS: TPR, FNR, FPR, TNR
        ===========================================================================
        """

        sens_conf = {}
        for i in labels:
            sens_conf[labels[i]] = confusion_matrix(list(sens_df[labels[i]]['t']), list(sens_df[labels[i]]['p']), labels=[0,1]).ravel()
        
        true_positive_rate = {}
        false_positive_rate = {}
        true_negative_rate = {}
        false_negative_rate = {}
        
        for i in labels:
            true_positive_rate[labels[i]] = (sens_conf[labels[i]][3]/(sens_conf[labels[i]][3]+sens_conf[labels[i]][2]))
            false_positive_rate[labels[i]] = (sens_conf[labels[i]][1]/(sens_conf[labels[i]][1]+sens_conf[labels[i]][0]))
            true_negative_rate[labels[i]] = 1 - false_positive_rate[labels[i]]
            false_negative_rate[labels[i]] = 1 - true_positive_rate[labels[i]]
       
        return(true_positive_rate,false_positive_rate,true_negative_rate,false_negative_rate)



    def ability_plots(self, labels,TPR,FPR,TNR,FNR):
        """
        METHOD: ability_plots
        ======================

        DESCRIPTION: Method to plot bars with each of the rates for positives
        and negatives as previously obtained in ability
        ===========================================================================

        PARAMETERS: labels,TPR(True Positive Rate),FPR(True Positive Rate),
        TNR(True Negative Rate) ,FNR(False Negative Rate)
        ===========================================================================

        RETURNS: A bar plot of the ability disparties of each TPR, FPR,TNR,FNR 
        based on the senstive variable
        ==========================================================================
        """

        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=("True Positive Rate", "False Positive Rate", "True Negative Rate", "False Negative Rate"))

        x_axis = list(labels.values())
        fig.add_trace(
            go.Bar(x = x_axis, y=list(TPR.values())),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(FPR.values())),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(TNR.values())),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(FNR.values())),
            row=2, col=2
        )

        fig.update_layout(showlegend=False,height=400, width=800, title_text="Ability Disparities")
        fig.show()

    def ability_metrics(self, TPR,FPR,TNR,FNR):
        """
        METHOD: ability_metrics
        ======================

        DESCRIPTION: Method to test the null hypothesis between TPR, FPR, TNR and FNR
        respectively with a given p value
        ===========================================================================

        PARAMETERS: TPR(True Positive Rate),FPR(True Positive Rate),
        TNR(True Negative Rate) ,FNR(False Negative Rate)
        ===========================================================================

        RETURNS: A message accepting or rejecting the null hypothesis, with the 
        respective p-values 
        ===========================================================================
        """

        TPR_p = chisquare(list(np.array(list(TPR.values()))*100))[1]
        FPR_p = chisquare(list(np.array(list(FPR.values()))*100))[1]
        TNR_p = chisquare(list(np.array(list(TNR.values()))*100))[1]
        FNR_p = chisquare(list(np.array(list(FNR.values()))*100))[1]
        

        if TPR_p <= 0.01:
            print("*** Reject H0: Significant True Positive Disparity with p=",TPR_p)
        elif TPR_p <= 0.05:
            print("** Reject H0: Significant True Positive Disparity with p=",TPR_p)
        elif TPR_p <= 0.1:
            print("*  Reject H0: Significant True Positive Disparity with p=",TPR_p)
        else:
            print("Accept H0: True Positive Disparity Not Detected. p=",TPR_p)

        if FPR_p <= 0.01:
            print("*** Reject H0: Significant False Positive Disparity with p=",FPR_p)
        elif FPR_p <= 0.05:
            print("** Reject H0: Significant False Positive Disparity with p=",FPR_p)
        elif FPR_p <= 0.1:
            print("*  Reject H0: Significant False Positive Disparity with p=",FPR_p)
        else:
            print("Accept H0: False Positive Disparity Not Detected. p=",FPR_p)

        if TNR_p <= 0.01:
            print("*** Reject H0: Significant True Negative Disparity with p=",TNR_p)
        elif TNR_p <= 0.05:
            print("** Reject H0: Significant True Negative Disparity with p=",TNR_p)
        elif TNR_p <= 0.1:
            print("*  Reject H0: Significant True Negative Disparity with p=",TNR_p)
        else:
            print("Accept H0: True Negative Disparity Not Detected. p=",TNR_p)

        if FNR_p <= 0.01:
            print("*** Reject H0: Significant False Negative Disparity with p=",FNR_p)
        elif FNR_p <= 0.05:
            print("** Reject H0: Significant False Negative Disparity with p=",FNR_p)
        elif FNR_p <= 0.1:
            print("*  Reject H0: Significant False Negative Disparity with p=",FNR_p)
        else:
            print("Accept H0: False Negative Disparity Not Detected. p=",FNR_p)




    def predictive(self, labels,sens_df):
        """
        METHOD: Predictive Parity 
        ======================

        DESCRIPTION: This model shows the predictive Paritybased on the sensitive 
        group.
        ==========================================================================

        PARAMETERS: sens_df, labels
        ==========================================================================

        RETURNS: Predictive disparity based on the senstive variables and test the 
        Hypothesis H0: No Significant Predictive Disparity.
        precision_dic,fig,pred_p
        ==========================================================================
        """

        precision_dic = {}

        for i in labels:
            precision_dic[labels[i]] = precision_score(sens_df[labels[i]]['t'],sens_df[labels[i]]['p'])

        fig = go.Figure([go.Bar(x=list(labels.values()), y=list(precision_dic.values()))])
        
        pred_p = chisquare(list(np.array(list(precision_dic.values()))*100))[1]
        
        return(precision_dic,fig,pred_p)


    def identify_bias(self, sensitive,labels):
        """
        METHOD: identify_bias
        ======================

        DESCRIPTION: Identifies the bias by comparing the average of all attributes
        to one group vs to all groups in a graphical manner
        ===========================================================================

        PARAMETERS: The sensitive group and the labels
        ===========================================================================

        RETURNS: The representative Model, the ability plots and ability metrics and 
        predictive diparity
        ===========================================================================
        """
        try:
    
            predictions = self.model.predict(self.X_test)
            cont_table,sens_df,rep_fig,rep_p,classification,confusion,metric = self.representation(sensitive,labels,predictions)
        
            print("REPRESENTATION")
            rep_fig.show()

            print(cont_table,'\n')

            if rep_p <= 0.01:
                print("*** Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
            elif rep_p <= 0.05:
                print("** Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
            elif rep_p <= 0.1:
                print("* Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
            else:
                print("Accept H0: No Significant Relation Between",sensitive,"and Target Detected. p=",rep_p)

            TPR, FPR, TNR, FNR = self.ability(sens_df,labels)
            print("\n\nABILITY")
            self.ability_plots(labels,TPR,FPR,TNR,FNR)
            self.ability_metrics(TPR,FPR,TNR,FNR)


            precision_dic, pred_fig, pred_p = self.predictive(labels,sens_df)
            print("\n\nPREDICTIVE")
            pred_fig.update_layout(height=400, width=800)
            pred_fig.show()

            if pred_p <= 0.01:
                print("*** Reject H0: Significant Predictive Disparity with p=",pred_p)
            elif pred_p <= 0.05:
                print("** Reject H0: Significant Predictive Disparity with p=",pred_p)
            elif pred_p <= 0.1:
                print("* Reject H0: Significant Predictive Disparity with p=",pred_p)
            else:
                print("Accept H0: No Significant Predictive Disparity. p=",pred_p)

        except: 
            print("This is not correct,please enter the correct values")
    def understand_shap(self,labels,sensitive, affected_group,affected_target):
        """
        METHOD: understand_shap
        =======================

        DESCRIPTION: Identifies the bias by comparing the average of all attributes
        to one group vs to all groups in a graphical manner. The group is the
        previously identified as the affected one
        ===========================================================================

        PARAMETERS: The sensitive group, labels, affected group and affected target
        ===========================================================================

        RETURNS: A graph with mean sharp values, violin graph, bar graphs with the 
        average comparisom to true class members, average comparison to all members, 
        sharp waterfall and dependency scatter plot
        ==========================================================================
        """
        
        try: 
            
            import shap
            explainer = shap.Explainer(self.model)

            full_table = self.X_test.copy()
            full_table['t'] = self.y_test
            full_table['p'] = self.model.predict(self.X_test)
            full_table

            shap_values = explainer(self.X_test)
            sens_glob_coh = np.where(self.X_test[sensitive]==list(labels.keys())[0],labels[0],labels[1])

            misclass = full_table[full_table.t != full_table.p]
            affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]
            shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
            #sens_mis_coh = np.where(affected_class[sensitive]==list(labels.keys())[0],labels[0],labels[1])


            figure,axes = plt.subplots(nrows=2, ncols=2,figsize=(10,7))
            plt.subplots_adjust(right=1.4,wspace=1)

            print("Model Importance Comparison")
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            shap.plots.bar(shap_values.cohorts(sens_glob_coh).abs.mean(0),show=False)
            plt.subplot(1, 2, 2) # row 1, col 2 index 1
            shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
            shap.plots.bar(shap_values2)
            #shap.plots.bar(shap_values2)
            
            #Plots the violin graph 
            shap.summary_plot(shap_values2, plot_type='violin')

            full_table['t'] = self.y_test
            full_table['p'] = self.model.predict(self.X_test)
            #full_table=full_table[['checking_account','credit_amount','duration','sex','t','p']]

            misclass = full_table[full_table.t != full_table.p]
            affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

            truclass = full_table[full_table.t == full_table.p]
            tru_class = truclass[(truclass[sensitive] == affected_group) & (truclass.t == affected_target)]

            x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
            affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-tru_class.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

            #plt.figsize([10,10])
            #plt.bar(x_axis,affect_character)

            fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])

            print("Affected Attribute Comparison")
            print("Average Comparison to True Class Members")
            fig.show()

            misclass = full_table[full_table.t != full_table.p]
            affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

            #truclass = full_table[full_table.t == full_table.p]
            tru_class = full_table[(full_table[sensitive] == affected_group) & (full_table.p == affected_target)]

            affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-full_table.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

            #plt.figsize([10,10])
            #plt.bar(x_axis,affect_character)

            fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])
            print("Average Comparison to All Members")
            fig.show()

            print("Random Affected Decision Process")
            explainer = shap.Explainer(self.model)
            shap.plots.waterfall(explainer(affected_class.drop(['t','p'],axis=1))[randrange(0, len(affected_class))],show=False);
            # Dependency Scatter Plot 
            values = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
            shap.plots.scatter(shap_values2[:, values], color=shap_values2)

        except: 
            print("This is not correct,please enter the correct values")

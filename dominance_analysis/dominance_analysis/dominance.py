import math
from itertools import combinations
import logging
import numpy as np
from collections import defaultdict
import pandas as pd
import statsmodels.api as sm
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.plotting import figure, show
from plotly.offline import init_notebook_mode, iplot
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

output_notebook()
init_notebook_mode(connected=True)

OBJECTIVE_REGRESSION = 1
OBJECTIVE_CLASSIFICATION = 0

DATA_XY = 0
DATA_CORR_MAT = 1
DATA_COV_MAT = 2


def train_linear_model(model_name, data, x, y, weights):
    x = data[x]
    y = data[y]
    lin_reg = LinearRegression(copy_X=True)
    lin_reg.fit(x, y, sample_weight=weights)
    return model_name, lin_reg.score(x, y, sample_weight=weights)


class Dominance:
    """docstring for ClassName"""

    def __init__(self,
                 data, target,
                 sample_weight=None,
                 top_k_features=15,
                 objective=OBJECTIVE_REGRESSION,
                 pseudo_r2='mcfadden',
                 data_format=DATA_XY):
        """

        Args:
            data: Complete Dataset, should be a Pandas DataFrame.
            target: Name of the target variable, it should be present in passed dataset.
            top_k_features: Number of features to choose from all available features, or list of features
                to be used.
            objective: It can take value either 0 or 1. 0 for Classification and 1 for Regression.
                By default, the package will run for Regression.
            pseudo_r2: It can take one of the Pseudo R-Squared measures - "mcfadden","nagelkerke",
                "cox_and_snell" or "estrella", where default="mcfadden". It's not needed in case
                of regression (objective=1).
            data_format: It can take value 0, 1 or 2. 0 is for raw data, 1 is when correlation
                matrix (correlation of predictors with the target variable) is being passed,
                2 is when covraiance matrix (covariance of predictors with the the traget variable)
                is being passed. By default, the package will run for raw data (data_format=0).
                This parameter is not needed in case of classification.
        """
        self.data = data
        self.target = target
        self.objective = objective
        self.sample_weight = sample_weight

        self.data_format = data_format
        if self.data_format == DATA_XY:
            if (self.objective == OBJECTIVE_CLASSIFICATION):
                self.data['intercept'] = 1

            if isinstance(top_k_features, int):
                self.top_k_features = top_k_features if top_k_features else min((len(self.data.columns) - 1), 15)
                assert self.top_k_features > 1 and self.top_k_features < len(self.data.columns), \
                    "Value of top_k_features ranges from 1 to n-1 !"
            elif isinstance(top_k_features, list):
                self.top_k_features = top_k_features
                assert all(x in self.data.columns for x in top_k_features), \
                    "top_k_features must contain feature names!"
            else:
                raise TypeError('top_k must be integer or list of feature names!')
            self.pseudo_r2 = pseudo_r2

        self.incrimental_r2 = None
        self.percentage_incremental_r2 = None
        self.complete_dominance_stats = None
        self.model_rsquares = None
        self.variable_stats = None
        self.stats = None

        self.complete_model_rsquare()

    def conditional_dominance(self, model_rsquares, model_features_k, model_features_k_minus_1, columns):
        all_features = model_features_k[0]
        total_model_r2 = model_rsquares[" ".join(all_features)]

        interactional_comp_dom = {}
        for i in model_features_k_minus_1:
            interactional_comp_dom[" ".join(set(all_features) - set(i))] = {
                " ".join(set(i)): total_model_r2 - model_rsquares[" ".join(i)]}

        interactional_dominance = {
            " ".join(set(all_features) - set(i)): total_model_r2 - model_rsquares[" ".join(i)]
            for i in model_features_k_minus_1}
        return interactional_dominance, interactional_comp_dom

    def individual_dominance(self, model_rsquares, model_features, columns):
        return {" ".join(col): model_rsquares[" ".join(col)]
                for col in model_features}

    def partial_dominance(self, model_rsquares, model_features_k, model_features_k_minus_1, columns):
        pd = {col: [] for col in columns}
        model_features_k_set = [set(x) for x in model_features_k]
        model_features_k_minus_1_set = [set(x) for x in model_features_k_minus_1]

        for i, k_features in enumerate(model_features_k_set):
            for j, k_minus1_features in enumerate(model_features_k_minus_1_set):
                add_k = k_features - k_minus1_features
                if len(add_k) == 1:
                    pd[list(add_k)[0]].append(
                        model_rsquares[" ".join(model_features_k[i])] -
                        model_rsquares[" ".join(model_features_k_minus_1[j])])

        pd_comp_dom = {col: {} for col in columns}

        for i, k_features in enumerate(model_features_k_set):
            for j, k_minus1_features in enumerate(model_features_k_minus_1_set):
                add_k = k_features - k_minus1_features
                if len(add_k) == 1:
                    pd_comp_dom[list(add_k)[0]].update({
                        " ".join(model_features_k_minus_1[j]):
                            model_rsquares[" ".join(model_features_k[i])] -
                            model_rsquares[" ".join(model_features_k_minus_1[j])]
                    })

        return pd, pd_comp_dom

    def model_features_combination(self, columns):
        return [list(combinations(columns, i)) for i in range(1, len(columns) + 1)]

    def McFadden_RSquare(self, columns):
        cols = columns.copy()
        cols.append('intercept')
        # print("model columns :",cols)
        log_clf = sm.Logit(self.data[self.target], self.data[cols])
        # result=log_clf.fit(disp=0,method='powell')
        try:
            result = log_clf.fit(disp=0)
        except:
            result = log_clf.fit(disp=0, method='powell')
        mcfadden_rsquare = result.prsquared
        return mcfadden_rsquare

    def Nagelkerke_Rsquare(self, columns):
        cols = columns.copy()
        cols.append('intercept')
        log_clf = sm.Logit(self.data[self.target], self.data[cols])
        N = self.data.shape[0]
        # result=log_clf.fit(disp=0,method='powell')
        try:
            result = log_clf.fit(disp=0)
        except:
            result = log_clf.fit(disp=0, method='powell')
        llf = result.llf
        llnull = result.llnull
        lm = np.exp(llf)
        lnull = np.exp(llnull)
        naglkerke_rsquare = (1 - (lnull / lm) ** (2 / N)) / (1 - lnull ** (2 / N))
        return naglkerke_rsquare

    def Cox_and_Snell_Rsquare(self, columns):
        cols = columns.copy()
        cols.append('intercept')
        log_clf = sm.Logit(self.data[self.target], self.data[cols])
        N = self.data.shape[0]
        # result=log_clf.fit(disp=0,method='powell')
        try:
            result = log_clf.fit(disp=0)
        except:
            result = log_clf.fit(disp=0, method='powell')
        llf = result.llf
        llnull = result.llnull
        lm = np.exp(llf)
        lnull = np.exp(llnull)
        cox_and_snell_rsquare = (1 - (lnull / lm) ** (2 / N))
        return cox_and_snell_rsquare

    def Estrella(self, columns):
        cols = columns.copy()
        cols.append('intercept')
        log_clf = sm.Logit(self.data[self.target], self.data[cols])
        N = self.data.shape[0]
        # result=log_clf.fit(disp=0,method='powell')
        try:
            result = log_clf.fit(disp=0)
        except:
            result = log_clf.fit(disp=0, method='powell')
        llf = result.llf
        llnull = result.llnull
        estrella_rsquare = 1 - ((llf / llnull) ** (-(2 / N) * llnull))
        return estrella_rsquare

    def Adjusted_McFadden_RSquare(self, columns):
        log_clf = sm.Logit(self.data[self.target], self.data[cols])
        # result=log_clf.fit(disp=0,method='powell')
        try:
            result = log_clf.fit(disp=0)
        except:
            result = log_clf.fit(disp=0, method='powell')
        llf = result.llf
        llnull = result.llnull
        adjusted_mcfadden_rsquare = 1 - ((llf - len(columns)) / llnull)
        return adjusted_mcfadden_rsquare

    def model_stats(self):
        # columns=list(self.data.columns.values)
        # columns.remove(self.target)
        # # print("Independent Variables : ",columns)

        if self.model_rsquares is not None:
            return self.model_rsquares

        columns = self.get_top_k()

        model_combinations = self.model_features_combination(columns)
        model_rsquares = {}
        if self.objective == OBJECTIVE_REGRESSION:
            with Pool(processes=cpu_count()) as pool:
                results = []
                for features_model_sizes in model_combinations:
                    for x in features_model_sizes:
                        x = list(x)
                        model_name = ' '.join(x)

                        results.append(pool.apply_async(
                            train_linear_model, (model_name, self.data, x, self.target, self.sample_weight)))
                for async_result in tqdm(results, desc='Waiting for results'):
                    model_name, r2 = async_result.get()
                    model_rsquares[model_name] = r2
        else:
            for i in tqdm(model_combinations):
                for j in i:
                    train_features = list(j)
                    if (self.pseudo_r2 == 'mcfadden'):
                        # print("mcfadden")
                        r_squared = self.McFadden_RSquare(train_features)
                    # elif(self.pseudo_r2=='adjusted_mcfadden'):
                    # 	r_squared=self.Adjusted_McFadden_RSquare(train_features)
                    elif (self.pseudo_r2 == 'nagelkerke'):
                        # print("nagelkerke")
                        r_squared = self.Nagelkerke_Rsquare(train_features)
                    elif (self.pseudo_r2 == 'cox_and_snell'):
                        r_squared = self.Cox_and_Snell_Rsquare(train_features)
                    elif (self.pseudo_r2 == 'estrella'):
                        r_squared = self.Estrella(train_features)
                    model_rsquares[" ".join(train_features)] = r_squared

        self.model_rsquares = model_rsquares
        return self.model_rsquares

    def incremental_rsquare(self):
        if self.incrimental_r2 is not None:
            return self.incrimental_r2

        if self.data_format == DATA_XY:
            columns = self.get_top_k()
            logging.info("Selected Predictors: %s", ", ".join(columns))
            logging.info("Creating models for %s possible combinations of %s features.",
                         2 ** len(columns) - 1, len(columns))
            model_rsquares = self.model_stats()
        else:
            if self.data_format == DATA_COV_MAT:
                columns = list(self.data.columns.values)
                d = np.sqrt(self.data.values.diagonal())
                corr_array = ((self.data.values.T / d).T) / d
                self.data = pd.DataFrame(data=corr_array, index=columns)
                self.data.columns = columns
            model_rsquares = self.Dominance_correlation()
            columns = list(self.data.columns.values)
            columns.remove(self.target)

        logging.info("Model Training Done.")

        logging.info("Calculating Variable Dominances.")
        variable_stats = self.variable_statistics(model_rsquares, columns)

        logging.info("Variable Dominance Calculation Done.")

        incrimental_r2 = {}
        for col in columns:
            l = [variable_stats[col]['individual_dominance'], variable_stats[col]['conditional_dominance']]
            l.extend(variable_stats[col]['partial_dominance'])
            incrimental_r2[col] = np.mean(l)

        self.incrimental_r2 = incrimental_r2
        self.percentage_incremental_r2 = {
            col: incrimental_r2[col] / np.sum(list(incrimental_r2.values())) for col in
            columns}

        return incrimental_r2

    def variable_statistics(self, model_rsquares, columns):
        if self.variable_stats is not None:
            return self.variable_stats

        stats = {}
        complete_dominance_stats = {}

        model_combinations_by_size = defaultdict(list)
        for i in self.model_features_combination(columns):
            for j in i:
                model_combinations_by_size[len(i)].append(i)

        for k in tqdm(range(len(columns), 1, -1), desc='Dominance stats'):
            model_features_k = model_combinations_by_size[k]
            model_features_k_minus_1 = model_combinations_by_size[k - 1]
            if k == len(columns):
                inter_dom, inter_comp_dom = self.conditional_dominance(
                    model_rsquares, model_features_k, model_features_k_minus_1, columns)
                stats['conditional_dominance'] = inter_dom
                complete_dominance_stats['interactional_dominance'] = inter_comp_dom
            else:
                part_dom, part_comp_dom = self.partial_dominance(
                    model_rsquares, model_features_k, model_features_k_minus_1, columns)
                stats['partial_dominance_%d' % k] = part_dom
                complete_dominance_stats['partial_dominance_%d' % k] = part_comp_dom

            if k == 2:
                ind_dom = self.individual_dominance(
                    model_rsquares, model_features_k_minus_1, columns)
                stats['individual_dominance'] = ind_dom
                complete_dominance_stats['individual_dominance'] = ind_dom

        variable_stats = {}
        for col in columns:
            variable_stats[col] = {
                'conditional_dominance': stats['conditional_dominance'][col],
                'individual_dominance': stats['individual_dominance'][col],
                'partial_dominance': [
                    np.mean(stats["partial_dominance_%d" % i][col])
                    for i in range(len(columns) - 1, 1, -1)]
            }

        self.stats = stats
        self.complete_dominance_stats = complete_dominance_stats
        self.variable_stats = variable_stats
        return self.variable_stats

    def dominance_stats(self):
        tf = pd.DataFrame(self.variable_stats).T
        tf['Interactional Dominance'] = tf['conditional_dominance']
        tf['Average Partial Dominance'] = tf['partial_dominance'].apply(lambda x: np.mean(x))
        tf['partial_dominance_count'] = tf['partial_dominance'].apply(lambda x: len(x))
        tf['Total Dominance'] = (tf['partial_dominance_count'] * tf['Average Partial Dominance'] + tf[
            'conditional_dominance'] + tf['individual_dominance']) / (tf['partial_dominance_count'] + 2)
        tf['Percentage Relative Importance'] = (tf['Total Dominance'] * 100) / tf['Total Dominance'].sum()
        tf = tf[['Interactional Dominance', 'individual_dominance', 'Average Partial Dominance', 'Total Dominance',
                 'Percentage Relative Importance']].sort_values("Total Dominance", ascending=False)
        tf.rename(columns={"individual_dominance": "Individual Dominance"}, inplace=True)
        return tf

    def get_top_k(self):
        if isinstance(self.top_k_features, list):
            return self.top_k_features

        columns = [x for x in self.data.columns.values if x != self.target]
        # remove intercept from top_k
        if self.objective == OBJECTIVE_REGRESSION:
            top_k_vars = SelectKBest(f_regression, k=self.top_k_features)
            top_k_vars.fit_transform(self.data[columns], self.data[self.target])
        else:
            columns.remove('intercept')
            try:
                top_k_vars = SelectKBest(chi2, k=self.top_k_features)
                top_k_vars.fit_transform(self.data[columns], self.data[self.target])
            except:
                top_k_vars = SelectKBest(f_classif, k=self.top_k_features)
                top_k_vars.fit_transform(self.data[columns], self.data[self.target])

        top_k = [columns[i] for i in top_k_vars.get_support(indices=True)]
        self.top_k_features = top_k
        return top_k

    def plot_waterfall_relative_importance(self, incremental_rsquare_df):
        index = list(incremental_rsquare_df['Features'].values)
        data = {'Percentage Relative Importance': list(incremental_rsquare_df['percentage_incremental_r2'].values)}
        df = pd.DataFrame(data=data, index=index)

        net = df['Percentage Relative Importance'].sum()
        # print("Net ",net)

        df['running_total'] = df['Percentage Relative Importance'].cumsum()
        df['y_start'] = df['running_total'] - df['Percentage Relative Importance']

        df['label_pos'] = df['running_total']

        df_net = pd.DataFrame.from_records([(net, net, 0, net)],
                                           columns=['Percentage Relative Importance', 'running_total', 'y_start',
                                                    'label_pos'], index=["net"])

        df = df.append(df_net)

        df['color'] = '#1de9b6'
        df.loc[df['Percentage Relative Importance'] == 100, 'color'] = '#29b6f6'
        df.loc[df['Percentage Relative Importance'] < 0, 'label_pos'] = df.label_pos - 10000
        df["bar_label"] = df["Percentage Relative Importance"].map('{:,.1f}'.format)

        TOOLS = "reset,save"
        source = ColumnDataSource(df)
        p = figure(tools=TOOLS, x_range=list(df.index), y_range=(0, net + 10),
                   plot_width=1000, title="Percentage Relative Importance Waterfall")

        p.segment(x0='index', y0='y_start', x1="index", y1='running_total',
                  source=source, color="color", line_width=35)

        p.grid.grid_line_alpha = 0.4
        p.yaxis[0].formatter = NumeralTickFormatter(format="(0 a)")
        p.xaxis.axis_label = "Predictors"
        p.yaxis.axis_label = "Percentage Relative Importance(%)"
        p.xaxis.axis_label_text_font_size = '12pt'
        p.yaxis.axis_label_text_font_size = '12pt'

        labels = LabelSet(x='index', y='label_pos', text='bar_label',
                          text_font_size="11pt", level='glyph',
                          x_offset=-14, y_offset=0, source=source)
        p.add_layout(labels)
        p.xaxis.major_label_orientation = -math.pi / 4
        show(p)

    def plot_incremental_rsquare(self):
        incremental_rsquare_df1 = pd.DataFrame()
        incremental_rsquare_df1['Features'] = self.incrimental_r2.keys()
        incremental_rsquare_df1['incremental_r2'] = self.incrimental_r2.values()
        incremental_rsquare_df1.sort_values('incremental_r2', ascending=False, inplace=True)

        incremental_rsquare_df2 = pd.DataFrame()
        incremental_rsquare_df2['Features'] = self.percentage_incremental_r2.keys()
        incremental_rsquare_df2['percentage_incremental_r2'] = self.percentage_incremental_r2.values()
        incremental_rsquare_df2.sort_values('percentage_incremental_r2', ascending=False, inplace=True)

        incremental_rsquare_df = pd.merge(left=incremental_rsquare_df1, right=incremental_rsquare_df2)
        incremental_rsquare_df['percentage_incremental_r2'] = incremental_rsquare_df['percentage_incremental_r2'] * 100
        # Bala changes start
        if (self.data_format == 0):
            iplot(incremental_rsquare_df[['Features', 'incremental_r2']].set_index("Features").iplot(
                asFigure=True,
                kind='bar',
                title="Incremetal " + ("Pseudo " if self.objective != 1 else " ") +
                      "R Squared for Top " + str(self.top_k_features) + " Variables ",
                yTitle="Incremental R2",
                xTitle="Estimators"))
            iplot(incremental_rsquare_df[['Features', 'percentage_incremental_r2']].iplot(
                asFigure=True, kind='pie',
                title="Percentage Relative Importance for Top " + str(self.top_k_features) + " Variables ",
                values="percentage_incremental_r2",
                labels="Features"))
        else:
            iplot(incremental_rsquare_df[['Features', 'incremental_r2']].set_index("Features").iplot(
                asFigure=True,
                kind='bar',
                title="Incremetal " + ("Pseudo " if self.objective != 1 else " ") + "R Squared of " + " Variables ",
                yTitle="Incremental R2",
                xTitle="Estimators"))
            iplot(incremental_rsquare_df[['Features', 'percentage_incremental_r2']].iplot(
                asFigure=True, kind='pie',
                title="Percentage Relative Importance of " + " Variables ",
                values="percentage_incremental_r2",
                labels="Features"))
        # Bala changes end#Bala changes end
        self.plot_waterfall_relative_importance(incremental_rsquare_df[['Features', 'percentage_incremental_r2']])

    def predict_general_dominance(self):
        general_dominance = []
        l = list(self.dominance_stats().index)
        for index, i in enumerate(l):
            general_dominance.append({"Predictors": i, "Dominating": l[index + 1:]})

        return pd.DataFrame(general_dominance)[['Predictors', 'Dominating']]

    def predict_conditional_dominance(self):

        general_dominance = []
        l = list(self.dominance_stats().index)
        for index, i in enumerate(l):
            general_dominance.append({"Predictors": i, "Dominating": l[index + 1:]})

        conditinal_dominance = []
        for x in general_dominance:
            predictor = x['Predictors']
            cd = True
            if (len(x['Dominating']) > 0):
                for j in x['Dominating']:
                    if ((self.variable_stats[predictor]['individual_dominance'] < self.variable_stats[j][
                        'individual_dominance']) or (
                            self.variable_stats[predictor]['conditional_dominance'] < self.variable_stats[j][
                        'conditional_dominance'])):
                        cd = False
                        break

                    if (cd):
                        for index, i in enumerate(self.variable_stats[predictor]['partial_dominance']):
                            if (i < self.variable_stats[j]['partial_dominance'][index]):
                                cd = False
                                break
            else:
                cd = False

            if (cd):
                conditinal_dominance.append(
                    {"Predictors": predictor, "Conditional Dominance": True, "Dominating": x['Dominating']})
            else:
                conditinal_dominance.append(
                    {"Predictors": predictor, "Conditional Dominance": False, "Dominating": None})

        return pd.DataFrame(conditinal_dominance)[['Predictors', 'Conditional Dominance', 'Dominating']]

    def predict_complete_dominance(self):
        conditional_dominance_df = self.predict_conditional_dominance()
        conditional_dominant_predictors = list(
            conditional_dominance_df[conditional_dominance_df['Conditional Dominance'] == True]['Predictors'].values)
        predictors = list(conditional_dominance_df['Predictors'].values)

        # print(conditional_dominant_predictors,predictors)

        cd_df = []

        cds = self.complete_dominance_stats
        # print(conditional_dominant_predictors)

        for i in conditional_dominant_predictors:
            # print(i,conditional_dominance_df)
            dominating = conditional_dominance_df[conditional_dominance_df['Predictors'] == i]['Dominating'].values[0]
            complete_dominance = True
            for j in [p for p in list(cds.keys()) if p != 'interactional_dominance']:
                if (j == 'individual_dominance'):
                    if (sum(cds[j][i] > [cds[j][key] for key in dominating]) != len(dominating)):
                        complete_dominance = False
                        break
                else:
                    search_index = []
                    for k in dominating:
                        if (complete_dominance):
                            for key in cds[j][i].keys():
                                l = list(set(predictors) - set(key.split(" ")) - set([i]))
                                [search_index.append((i, key, c)) for c in l]

                    search_index = list(set(search_index))

                    if (complete_dominance):
                        for search in search_index:
                            # print(search[0],search[1],search[2],cds[j][search[0]][search[1]],cds[j][search[2]][search[1]])
                            if (cds[j][search[0]][search[1]] < cds[j][search[2]][search[1]]):
                                complete_dominance = False
                                break

            cd_df.append({"Predictors": i, "Dominating": dominating})

        return pd.DataFrame(cd_df)[['Predictors', 'Dominating']]

    def dominance_level(self):
        gen_dom = self.predict_general_dominance()
        condition_dom = self.predict_conditional_dominance()
        comp_dom = self.predict_complete_dominance()

        gen_dom.rename(columns={'Dominating': 'Generally Dominating'}, inplace=True)
        condition_dom.drop('Conditional Dominance', inplace=True, axis=1)
        condition_dom.rename(columns={'Dominating': 'Conditionally Dominating'}, inplace=True)
        comp_dom.rename(columns={'Dominating': 'Completelly Dominating'}, inplace=True)

        return pd.merge(
            pd.merge(left=gen_dom, right=condition_dom[['Predictors', 'Conditionally Dominating']], how='left'),
            comp_dom, how='left').fillna("")

    def complete_model_rsquare(self):
        if self.data_format == 0:  # Bala changes
            print("Selecting %s Best Predictors for the Model" % self.top_k_features)
            columns = self.get_top_k()
            print("Selected Predictors : ", columns)
            print()

            if self.objective == OBJECTIVE_REGRESSION:
                print("*" * 20, " R-Squared of Complete Model : ", "*" * 20)
                lin_reg = LinearRegression()
                lin_reg.fit(self.data[columns], self.data[self.target], sample_weight=self.sample_weight)
                r_squared = lin_reg.score(self.data[columns], self.data[self.target], sample_weight=self.sample_weight)
                if self.sample_weight is not None:
                    print("Weighted R2: %s" % (r_squared))
                else:
                    print("R2: %s" % (r_squared))
                print()
            else:
                print("*" * 20, " Pseudo R-Squared of Complete Model : ", "*" * 20)
                print()

                if (self.pseudo_r2 == 'mcfadden'):
                    print("MacFadden's R-Squared : %s " % (self.McFadden_RSquare(columns)))
                elif (pseudo_r2 == 'nagelkerke'):
                    print("Nagelkerke R-Squared : %s " % (self.Nagelkerke_Rsquare(columns)))
                elif (pseudo_r2 == 'cox_and_snell'):
                    print("Cox and Snell R-Squared : %s " % (self.Cox_and_Snell_Rsquare(columns)))
                else:
                    print("Estrella R-Squared : %s " % (self.Estrella(columns)))
                print()
        # Bala changes start
        else:
            if (self.data_format == 2):
                columns = list(self.data.columns.values)
                d = np.sqrt(self.data.values.diagonal())
                corr_array = ((self.data.values.T / d).T) / d
                self.data = pd.DataFrame(data=corr_array, index=columns)
                self.data.columns = columns
                print()
            columns = list(self.data.columns.values)
            columns.remove(self.target)
            corr_all_matrix = self.data.loc[columns, [self.target]]
            corr_pred_matrix = self.data.loc[columns, columns]
            corr_pred_matrix_inverse = pd.DataFrame(np.linalg.pinv(corr_pred_matrix.values), corr_pred_matrix.columns,
                                                    corr_pred_matrix.index)
            beta = corr_pred_matrix_inverse.dot(corr_all_matrix)
            corr_all_matrix_transpose = corr_all_matrix.transpose()
            r_squared = corr_all_matrix_transpose.dot(beta)
            print("R Squared : %s" % (r_squared.iloc[0, 0]))
            print()

    # Bala changes ends

    # Bala changes starts
    def Dominance_correlation(self):
        ## Calculating Incremental R2 from Correlation Matrix
        columns = list(self.data.columns.values)
        columns.remove(self.target)
        print("Predictors : ", columns)
        print()

        print("Calculating R2 for %s possible combinations of %s features :" % ((2 ** len(columns)) - 1, len(columns)))
        model_combinations = self.model_features_combination(columns)
        model_rsquares = {}
        for i in tqdm(model_combinations):
            for j in i:
                model_combinations = list(j)
                corr_all_matrix = self.data.loc[model_combinations, [self.target]]
                corr_pred_matrix = self.data.loc[model_combinations, model_combinations]
                corr_pred_matrix_inverse = pd.DataFrame(np.linalg.pinv(corr_pred_matrix.values),
                                                        corr_pred_matrix.columns, corr_pred_matrix.index)
                beta = corr_pred_matrix_inverse.dot(corr_all_matrix)
                corr_all_matrix_transpose = corr_all_matrix.transpose()
                r_square = corr_all_matrix_transpose.dot(beta)
                model_rsquares[" ".join(model_combinations)] = r_square.iloc[0, 0]
        self.model_rsquares = model_rsquares
        return self.model_rsquares


# Bala changes ends

class Dominance_Datasets:
    """docstring for Dominance_Datasets"""

    @classmethod
    def get_breast_cancer(cls):
        print(
            """The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is downloaded from: https://goo.gl/U2Uwz2""")
        print("""Internally using load_breast_cancer function from sklearn.datasets """)
        breast_cancer_data = pd.DataFrame(data=load_breast_cancer()['data'],
                                          columns=load_breast_cancer()['feature_names'])
        breast_cancer_data['target'] = load_breast_cancer()['target']
        target_dict = dict(
            {j for i, j in zip(load_breast_cancer()['target_names'], enumerate(load_breast_cancer()['target_names']))})
        breast_cancer_data['target_names'] = breast_cancer_data['target'].map(target_dict)
        return breast_cancer_data.iloc[:, :-1]

    @classmethod
    def get_boston(cls):
        print(
            """The copy of Boston Housing Dataset is downloaded from: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html""")
        print("""Internally using load_boston function from sklearn.datasets """)
        boston_data = pd.DataFrame(data=load_boston()['data'], columns=load_boston()['feature_names'])
        boston_data['House_Price'] = load_boston()['target']
        return boston_data

    def __init__(self):
        print("Datasets for Dominance Analysis")

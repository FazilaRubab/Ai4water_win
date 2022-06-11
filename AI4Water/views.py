import site
site.addsitedir("E:\\Ai4water latest\\ai4waterlatest")
from django.shortcuts import render, redirect,HttpResponse
import json
import pandas as pd
import numpy as np
from io import StringIO
# from pyai4water.preprocessing.transformations import Transformations
from ai4water.datasets import busan_beach
from ai4water import Model
from ai4water.eda import EDA
from ai4water.preprocessing import DataSet,Transformation
from ai4water.postprocessing import explain
from ai4water.postprocessing.explain import explain_model_with_lime
from lime import lime_tabular
import shap
from sklearn.model_selection import train_test_split
from SeqMetrics import RegressionMetrics
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer
import math
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('Agg')
aiModal = None
PREFIX = f"hpo_{dateandtime_now()}"
ITER = 0

def index(request):
    return render(request, "index.html")


def renderModel(request):
    return render(request, "Model.html")

def loadEDA(request):
    msg = ""
    showData = False
    dataToShow = None
    if request.POST:
        # print("EDA Type", request.POST.get('type'))
        formData = dict(request.POST.items())
        # print(formData)
        if formData['csvFile'] != "" and formData['csvFile'] != None:
            file = formData['csvFile']
            csvStringIO = StringIO(file)
            df = pd.read_csv(csvStringIO, sep=",", header=0)
            inputColumns = request.POST.get("inputFeatures", "")
            outputColumns = request.POST.get("outputFeatures", "")
            inputColumns = inputColumns.split(", ")
            outputColumns = outputColumns.split(", ")
            inputColumns = [x.rstrip() for x in inputColumns]
            outputColumns = [x.rstrip() for x in outputColumns]
            dpi = 300
            selectedColumnsData = pd.concat([df[inputColumns], df[outputColumns]], axis=1)
            # print("selectedColumnsData", selectedColumnsData)
            if formData['dpi'] != None and formData['dpi'] != '':
                dpi = int(formData['dpi'])
            transformer = Transformation(method='minmax')
            transformedData = transformer.fit_transform(data=selectedColumnsData)
            # print(transformedData)
            # print("transformedData" , transformedData)
            eda = EDA(data=transformedData, in_cols=inputColumns, out_cols=outputColumns,
                      dpi=dpi,
                      save=formData['save'])
            # print('type', formData['type'])
            if formData['type'] == "heat map":
                eda.heatmap()
            if formData['type'] == "prob":
                eda.probability_plots()

            if formData['type'] == "stats":
                print("STATS")
                stats = eda.stats()
                print("EDA Stats", stats)
                showData = True
                if stats != None:
                    # dataToShow =pd.DataFrame.from_dict(stats)
                    # dataToShow = dataToShow.to_json(orient='records')
                    dataToShow = json.dumps(stats)

            if formData['type'] == "pcs":
                eda.plot_pcs()

            if formData['type'] == "missing":
                missingPlot = eda.plot_missing()
                if missingPlot == None:
                    msg = "No missing values found in the dataset"

            if formData['type'] == "index":
                eda.plot_index()

            if formData['type'] == "histogram":
                eda.plot_histograms()

            if formData['type'] == "ecdf":
                eda.plot_ecdf()

            if formData['type'] == "data":
                eda.plot_data(subplots=True, max_cols_in_plot=20, figsize=(14, 20),
                              ignore_datetime_index=True)

            if formData['type'] == "lag":
                eda.lag_plot()

            if formData['type'] == "partial a cor":
                eda.partial_autocorrelation(n_lags=15)

            if formData['type'] == "partial a cor":
                eda.partial_autocorrelation(n_lags=15)

            if formData['type'] == "grouped scatter":
                eda.grouped_scatter()

            if formData['type'] == "corelation":
                eda.correlation()

            if formData['type'] == "auto corelation":
                eda.autocorrelation()

            if formData['type'] == "box":
                eda.box_plot()

        else:
            msg = "Please select input file from Data Collection section"
        return HttpResponse(
            json.dumps({'msg': msg, 'showData': showData, 'dataToShow': dataToShow}),
            content_type="application/json"
        )
def renderEDA(request):
    # msg = None
    # if request.POST:
    #     print("EDA Type" , request.POST.get('type'))
    #     formData = dict(request.POST.items())
    #     # print(formData)
    #     if formData['csvFile'] != "" and formData['csvFile'] != None:
    #         file = formData['csvFile']
    #         csvStringIO = StringIO(file)
    #         df = pd.read_csv(csvStringIO, sep=",", header=0)
    #         # print("file: ")
    #         # print(df)
    #         inputColumns = request.POST.get("inputFeatures", "")
    #         outputColumns = request.POST.get("outputFeatures", "")
    #         inputColumns = inputColumns.split(", ")
    #         outputColumns = outputColumns.split(", ")
    #         inputColumns = [x.rstrip() for x in inputColumns]
    #         outputColumns = [x.rstrip() for x in outputColumns]
    #         # print(formData)
    #         # file = request.POST.get("csvFile", "")
    #         # csvStringIO = StringIO(file)
    #         # df = pd.read_csv(csvStringIO, sep=",", header=0, index_col="index")
    #         # print(formData)
    #         dpi = 300
    #         selectedColumnsData = pd.concat([df[inputColumns], df[outputColumns]], axis=1)
    #         # print("selectedColumnsData", selectedColumnsData)
    #         if formData['dpi'] != None and formData['dpi'] != '':
    #             dpi = int(formData['dpi'])
    #         transformer = Transformation(method='minmax')
    #         transformedData = transformer.fit_transform(data=selectedColumnsData)
    #         # print("transformedData" , transformedData)
    #         eda = EDA(data=transformedData, in_cols=inputColumns, out_cols=outputColumns, dpi=dpi, save=formData['save'])
    #         # print('type', formData['type'])
    #         if formData['type'] == "heat map":
    #             eda.heatmap()
    #         if formData['type'] == "prob":
    #             eda.probability_plots()
    #
    #         if formData['type'] == "stats":
    #             print("STATS")
    #             stats = eda.stats()
    #             print("EDA Stats" , stats)
    #
    #
    #         if formData['type'] == "pcs":
    #             eda.plot_pcs()
    #
    #         if formData['type'] == "missing":
    #             missingPlot = eda.plot_missing()
    #             if missingPlot == None:
    #                 msg = "No missing values found in the dataset"
    #
    #         if formData['type'] == "index":
    #             eda.plot_index()
    #
    #         if formData['type'] == "histogram":
    #             eda.plot_histograms()
    #
    #         if formData['type'] == "ecdf":
    #             eda.plot_ecdf()
    #
    #         if formData['type'] == "data":
    #             eda.plot_data(subplots=True, max_cols_in_plot=20, figsize=(14, 20),
    #                           ignore_datetime_index=True)
    #
    #         if formData['type'] == "lag":
    #             eda.lag_plot()
    #
    #         if formData['type'] == "partial a cor":
    #             eda.partial_autocorrelation(n_lags=15)
    #
    #         if formData['type'] == "partial a cor":
    #             eda.partial_autocorrelation(n_lags=15)
    #
    #         if formData['type'] == "grouped scatter":
    #             eda.grouped_scatter()
    #
    #         if formData['type'] == "corelation":
    #             eda.correlation()
    #
    #         if formData['type'] == "auto corelation":
    #             eda.autocorrelation()
    #
    #         if formData['type'] == "box":
    #             eda.box_plot()
    #
    #
    #
    #     else:
    #         print("CSV file does not exist")
    #         msg = "Please train a Model first using Fit Button"
    #         return redirect('index')
    return render(request, "EDA.html")


def renderExplanation(request):
    if request.POST:
        formData = dict(request.POST.items())
        # print(formData)
        if formData['csvFile'] != "" and formData['csvFile'] != None:
            if aiModal != None:
                file = formData['csvFile']
                csvStringIO = StringIO(file)
                df = pd.read_csv(csvStringIO, sep=",", header=0)
                # print("file: ")
                # print(df)
                inputColumns = request.POST.get("inputFeatures", "")
                outputColumns = request.POST.get("outputFeatures", "")
                inputColumns = inputColumns.split(", ")
                outputColumns = outputColumns.split(", ")
                inputColumns = [x.rstrip() for x in inputColumns]
                outputColumns = [x.rstrip() for x in outputColumns]
                selectedColumnsData = pd.concat([df[inputColumns], df[outputColumns]], axis=1)

                if formData['type'] == "shap explainer":
                    X, y = aiModal.training_data()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                    se = explain.ShapExplainer(aiModal,
                              data=df[inputColumns].values,
                              train_data=X_train,
                              feature_names=aiModal.input_features,
)

                    # se.decision_plot()
                    se()
                    #se.force_plot_all(show=True)
                if formData['type'] == "lime explainer":
                    print("lime explainer")

                    # explain_model_with_lime(aiModal)
                    lime_exp = explain.LimeExplainer(model=aiModal,
                                             train_data=aiModal.training_data()[0],
                                             data=aiModal.test_data()[0],
                                             mode="regression")
                    lime_exp_output = lime_exp.explain_example(0)
                    print("lime_exp" , lime_exp_output)


                if formData['type'] == "permutation importance":
                    x_val, y_val = aiModal.validation_data()

                    pimp = explain.PermutationImportance(
                        aiModal.predict,
                        x_val,
                        y_val.reshape(-1, ),
                        feature_names=aiModal.input_features,
                        save=False
                    )

                    # plot permutatin importance of each feature  as box-plot
                    pimp.plot_1d_pimp()

                if formData['type'] == "partial dependence":
                    x, _ = aiModal.training_data()
                    pdp = explain.PartialDependencePlot(aiModal.predict,
                                                x,
                                                aiModal.input_features,
                                                save=False,
                                                num_points=14)

                    for feature in inputColumns:
                        pdp.plot_1d(feature)


            else:
                msg = "Please train a Model first using Fit Button"
                return redirect('Model')
        else:
            print("CSV file does not exist")
            msg = "Please train a Model first using Fit Button"
            return redirect('index')
    return render(request, "Explanation.html")


def renderMLRegression(request):
    return render(request, "ML-Regression.html")



def renderMLClassification(request):
    return render(request, "ML-Classification.html")


def objective_fn(
        prefix=None,
        **suggestions)->float:
    """This function must build, train and evaluate the ML model.
    The output of this function will be minimized by optimization algorithm.
    """
    suggestions = jsonize(suggestions)
    global ITER

    # evaluate model
    t, p = aiModal.predict(data='validation', return_true=True, process_results=False)
    val_score = RegressionMetrics(t, p).r2_score()

    if not math.isfinite(val_score):
        val_score = 1.0

    # since the optimization algorithm solves minimization algorithm
    # we have to subtract r2_score from 1.0
    # if our validation metric is something like mse or rmse,
    # then we don't need to subtract it from 1.0
    val_score = 1.0 - val_score

    ITER += 1

    print(f"{ITER} {val_score}")

    return val_score


def renderProcessing(request):
    msg = ""
    showData = False
    dataToShow = None
    formData = dict(request.POST.items())
    file = request.POST.get("csvFile", "")
    csvStringIO = StringIO(file)
    df = pd.read_csv(csvStringIO, sep=",", header=0)
    print("file: ")
    print(df)
    print(df['Glucose'])
    inputColumns = request.POST.get("inputFeatures", "")
    outputColumns = request.POST.get("outputFeatures", "")
    inputColumns = inputColumns.split(", ")
    outputColumns = outputColumns.split(", ")
    inputColumns = [x.rstrip() for x in inputColumns]
    outputColumns = [x.rstrip() for x in outputColumns]

    print(inputColumns, outputColumns)

    selectedColumnsData = pd.concat([df[inputColumns], df[outputColumns]], axis=1)
    if formData['type'] == 'fit':
        dataToShow = selectedColumnsData
        dataToShow = dataToShow.to_json(orient='records')
        showData = False
        model = Model(model="RandomForestRegressor",
                      x_transformation="minmax",
                      cross_validator={'TimeSeriesSplit': {'n_splits': 5}},
                      train_fraction = 0.7
                      ,val_fraction = 0.3,
                      input_features = inputColumns,
                      output_features = outputColumns
                      )

        print(model.config)
        model.fit(data = selectedColumnsData)
        global aiModal
        aiModal = model
        msg = "Modal has been trained successfully"
    if formData['type'] == 'evaluate':
        if aiModal != None:
            evaluatedResult = aiModal.evaluate()
            print("evaluatedResult" , evaluatedResult)
            msg = "Model has been evaluated. Evaluation Result: {}".format(evaluatedResult)
            dataToShow = selectedColumnsData
            dataToShow = dataToShow.to_json(orient='records')
            showData = False
        else:
            msg = "Please train a Model first using Fit Button"
            showData = False

    if formData['type'] == 'predict':
        if aiModal != None:
            aiModal.predict()
            dataToShow = selectedColumnsData
            dataToShow = dataToShow.to_json(orient='records')
            showData = False
        else:
            msg = "Please train a Model first using Fit Button"
            showData = False


    if formData['type'] == 'Cross value Score':
        if aiModal != None:
            tssplit_score = aiModal.cross_val_score(data=selectedColumnsData)
            print("tssplit_score", tssplit_score)
            msg = "Cross value Score: {}".format(tssplit_score[0])
            dataToShow = selectedColumnsData
            dataToShow = dataToShow.to_json(orient='records')
            showData = False
        else:
            msg = "Please train a Model first using Fit Button"
            showData = False

    if formData['type'] == 'Optimize Transformations':
        if aiModal != None:
            transformer = Transformation(method='minmax')
            transformedData = transformer.fit_transform(data=selectedColumnsData)
            print(transformedData)
            msg = "Data has been transformed"
            dataToShow = transformedData
            dataToShow = dataToShow.to_json(orient='records')
            showData = True
        else:
            msg = "Please train a Model first using Fit Button"
            showData = False
    import site
    site.addsitedir("E:\\Ai4water latest\\ai4waterlatest")

    if formData['type'] == 'Optimize Hyper Parameters':
        if aiModal != None:

            t, p = aiModal.predict(data='validation', return_true=True, process_results=False)
            val_score = RegressionMetrics(t, p).r2_score()

            val_score = RegressionMetrics(t, p).r2_score()

            if not math.isfinite(val_score):
                val_score = 1.0

            print("val_score",val_score)

            num_samples = 10
            space = [
                # maximum number of trees that can be built
                Integer(low=100, high=5000, name='iterations', num_samples=num_samples),
                # Used for reducing the gradient step.
                Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=num_samples),
                # Coefficient at the L2 regularization term of the cost function.
                Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=num_samples),
                # arger the value, the smaller the model size.
                Real(low=0.1, high=10, name='model_size_reg', num_samples=num_samples),
                # percentage of features to use at each split selection, when features are selected over again at random.
                Real(low=0.1, high=0.95, name='rsm', num_samples=num_samples),
                # number of splits for numerical features
                Integer(low=32, high=1032, name='border_count', num_samples=num_samples),
                # The quantization mode for numerical features.  The quantization mode for numerical features.
                Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles',
                                        'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')
            ]
            x0 = [200, 0.01, 1.0, 1.0, 0.2, 64, "Uniform"]

            optimizer = HyperOpt(
                algorithm="bayes",
                objective_fn=objective_fn(),
                param_space=space,
                x0=x0,
                num_iterations=15,
                process_results=False,
                # opt_path=f"results{SEP}{PREFIX}",
                verbosity=0,
            )

            results = optimizer.fit()
            print(f"optimized parameters are \n{optimizer.best_paras()}")
            optimizer._plot_convergence(save=False)
            optimizer._plot_parallel_coords(figsize=(14, 8), save=False)

            print("Results: " , results)

            msg = "Data has been transformed"
            dataToShow = transformedData
            dataToShow = dataToShow.to_json(orient='records')
            showData = False
        else:
            msg = "Please train a Model first using Fit Button"
            showData = False


    print(dataToShow)
    if showData == False:
        dataToShow = None
    return HttpResponse(
        json.dumps({'msg':msg, 'showData': showData, 'dataToShow': dataToShow }),
        content_type="application/json"
    )

    # return render(request, "model.html", {'msg':msg})

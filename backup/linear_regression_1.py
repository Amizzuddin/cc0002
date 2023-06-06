def generate_linear_regression_model(
        dataframe: pd.DataFrame,
        response_feature: str,
        predictor_feature: str,
        #add observation type for labelling purposes and extract data
    ):

        group_list: List[str] = np.unique(dataframe[groupby_features].values).tolist()
        grouped_dataframe: DataFrameGroupBy = dataframe.groupby(groupby_features) 


        figure, axes = plt.subplots(1, 1, figsize = (30, 20), constrained_layout=True)

        for group_name in group_list:

            print(f"Linear regression for '{group_name}:'\n")

            group_dataframe: pd.DataFrame = grouped_dataframe.get_group(group_name)

            response_dataframe = pd.DataFrame(group_dataframe[response_feature])
            predictor_dataframe = pd.DataFrame(group_dataframe[predictor_feature])


            # FOR TESTING PURPOISES ONLY!
            # print(f"response_dataframe: \n{response_dataframe}")
            # print(f"predictors_dataframe: \n{predictors_dataframe}")

            # new_group_name = group_name.replace('/', '_')
            # predictors_dataframe.to_csv(f"{new_group_name}_predictors_data.csv")
            # response_dataframe.to_csv(f"{new_group_name}_response_data.csv")

            print(f"predictors empty row \n{predictor_dataframe.isnull().sum()}")
            print(f"response empty row \n{response_dataframe.isnull().sum()}")

            if response_dataframe.isnull().values.any():
                response_dataframe.fillna(0, inplace=True)

                # for testing purposes
                # new_group_name = group_name.replace('/', '_')
                # response_dataframe.to_csv(f"{new_group_name}_response_data_revised.csv")

            # due to NOTE1 will dchange datetyime ro ordinal
            # predictor_dataframe_ordinal = predictor_dataframe.index.map(pd.Timestamp.toordinal())
            # predictor_dataframe_ordinal = predictor_dataframe.apply(pd.Timestamp.toordinal)
            # print(f"predictor_dataframe_ordinal type: {type(predictor_dataframe_ordinal)},\npredictor_dataframe_ordinal:\n{predictor_dataframe_ordinal}")
            predictor_dataframe_ordinal = pd.DataFrame()
            predictor_dataframe_ordinal[predictor_feature]: pd.DataFrame = pd.to_datetime(predictor_dataframe[predictor_feature]).apply(lambda date: date.toordinal()) # this line works!
            # print(f"[ordinal] predictor_dataframe type: {type(predictor_dataframe_ordinal)},\npredictor_dataframe:\n{predictor_dataframe_ordinal}")

            # linear regression will have problems with categorical data!
            linear_regression = LinearRegression()
            linear_regression.fit(predictor_dataframe_ordinal, response_dataframe)

            # Coefficients of the Linear Regression line
            print('Intercept of Regression \t: b = ', linear_regression.intercept_)
            print('Coefficients of Regression \t: a = ', linear_regression.coef_)
            print()

            # NOITE1 TypeError: cannot perform __rmul__ with this index type: DatetimeArray
            predictor_regression_line: pd.DataFrame = predictor_dataframe
            prediction_regression_line: pd.DataFrame = linear_regression.intercept_ + linear_regression.coef_ * predictor_dataframe_ordinal
            # prediction_regression_line.columns[0] = observation_type
            print(f"[TYPE] predictor_regression_line: {type(predictor_regression_line)}, prediction_regression_line: {prediction_regression_line}")
            prediction_regression_line.info()

            # # Print the Coefficients against Predictors
            # print(pd.DataFrame(list(zip(dataframe.columns, linear_regression.coef_[0])), columns = ["Predictors", "Coefficients"]))
            # print()

            
            # reshaping for prediction
            # response_dataframe_series: pd.Series = response_dataframe.squeeze()
            # predictors_dataframe_series: pd.Series = predictors_dataframe.squeeze()
            # print(f"[TYPE] response_dataframe_series: {type(response_dataframe_series)}, predictors_dataframe_series: {type(predictors_dataframe_series)}")
            # print(f"response_dataframe_series:\n{response_dataframe_series},\npredictors_dataframe_series: {predictors_dataframe_series}")

            # response_dataframe_series_array: np.ndarray = response_dataframe_series.values
            # predictors_dataframe_series_array: np.ndarray = predictors_dataframe_series.values
            # print(f"[TYPE] response_dataframe_series_array: {type(response_dataframe_series_array)}, predictors_dataframe_series_array: {type(predictors_dataframe_series_array)}")
            # print(f"response_dataframe_series_array:\n{response_dataframe_series_array},\npredictors_dataframe_series_array: {predictors_dataframe_series_array}")

            # response_dataframe_series_array_reshape: np.ndarray = response_dataframe_series_array.reshape(-1, 1)
            # predictors_dataframe_series_array_reshape: np.ndarray = predictors_dataframe_series_array.reshape(-1, 1)
            # print(f"[TYPE] response_dataframe_series_array_reshape: {type(response_dataframe_series_array_reshape)}, predictors_dataframe_series_array_reshape: {type(predictors_dataframe_series_array_reshape)}")
            # print(f"response_dataframe_series_array_reshape:\n{response_dataframe_series_array_reshape},\predictors_dataframe_series_array_reshape: {predictors_dataframe_series_array_reshape}")

            # response_prediction = linear_regression.predict(predictors_dataframe_series_array_reshape)
            # print(f"response_prediction type: {type(response_prediction)}")
            # print(f"response_prediction: {response_prediction}")

            # Issues starts here ValueError: Expected 2D array, got scalar array instead: array=month.
            # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.





            # response_prediction = linear_regression.predict(predictor_feature)
            # response_prediction = pd.Series(linear_regression.predict(predictor_feature), index=response_dataframe.index)
            # print(f"response_prediction type: {type(response_prediction)}")

            # plot
            axes.set_xlabel(predictor_feature)
            axes.set_ylabel(response_feature)


            # sb.lineplot( 
            #     x = time_feature, 
            #     y = observation_type,
            #     data = group_dataframe, 
            #     ax = axes,
            #     label =f'{group_name}'
            # )

            # last_data: pd.DataFrame = group_dataframe.tail(1)
            # axes.text(x=last_data[time_feature], y=last_data[observation_type], s=group_name, va="center")


            sb.lineplot( 
                x = "month", 
                y = predictor_feature,
                # x = predictor_feature, 
                # y = predictor_feature,
                # y = observation_type,
                data = prediction_regression_line, 
                ax = axes,
                label =f'{group_name}'
            )

        #     f = plt.figure(figsize=(16, 8))
        #     plt.plot(prediction_regression_line, predictor_regression_line, 'r-', linewidth = 3)
            
        # plt.show()
        axes.legend(loc='best')
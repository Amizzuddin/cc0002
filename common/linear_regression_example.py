 # Extract Response and Predictors
    # y = pd.DataFrame(pkmndata["Total"])
    # X = pd.DataFrame(pkmndata[["HP", "Attack", "Defense"]])

    # from sklearn.model_selection import train_test_split

    # Split the Dataset into Train and Test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


    # Linear Regression using Train Data
    # linreg = LinearRegression()         # create the linear regression object
    # linreg.fit(X_train, y_train)        # train the linear regression model
    
    # Coefficients of the Linear Regression line
    # print('Intercept of Regression \t: b = ', linreg.intercept_)
    # print('Coefficients of Regression \t: a = ', linreg.coef_)
    # print()

    # Print the Coefficients against Predictors
    # pd.DataFrame(list(zip(X_train.columns, linreg.coef_[0])), columns = ["Predictors", "Coefficients"])


    # Predict the Total values from Predictors
    # y_train_pred = linreg.predict(X_train)
    # y_test_pred = linreg.predict(X_test)

    # # Plot the Predictions vs the True values
    # f, axes = plt.subplots(1, 2, figsize=(24, 12))
    # axes[0].scatter(y_train, y_train_pred, color = "blue")
    # axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
    # axes[0].set_xlabel("True values of the Response Variable (Train)")
    # axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
    # axes[1].scatter(y_test, y_test_pred, color = "green")
    # axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
    # axes[1].set_xlabel("True values of the Response Variable (Test)")
    # axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
    # plt.show()


    # Check the Goodness of Fit (on Train Data)
    # print("Goodness of Fit of Model \tTrain Dataset")
    # print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
    # print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
    # print()

    # # Check the Goodness of Fit (on Test Data)
    # print("Goodness of Fit of Model \tTest Dataset")
    # print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
    # print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
    # print()
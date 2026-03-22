import statsmodels.formula.api as smf


def steps_calories_regression(df):
    model = smf.ols('Calories ~ TotalSteps + C(Id)', data=df)
    results = model.fit()

    return results.summary()


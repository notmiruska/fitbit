import statsmodels.formula.api as smf

def run_regression(df):
    
    # Convert Id to categorical (factor)
    df['Id'] = df['Id'].astype('category')
    
    # Fit model
    model = smf.ols('Calories ~ TotalSteps + C(Id)', data=df).fit()
    
    print(model.summary())
    
    return model
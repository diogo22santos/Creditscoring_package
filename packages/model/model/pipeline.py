from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import preprocessors as pp

CATEGORICAL_VARS_WITH_NA =['City', 'EmploymentPosition']

NUMERICAL_VARS_WITH_NA =['DebtToIncome','Education','Gender','HomeOwnershipType',
                         'VerificationType','PreviousScore']

TEMPORAL_VARS = 'DateOfBirth'

HIGH_CARDINALITY_VARS = ['City', 'EmploymentPosition']

CATEGORICAL_VARS = ['City', 'Country', 'EmploymentPosition']

default_pipe = Pipeline(
    [
        ('categorical_imputer',
         pp.CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)),
        ('numerical_inputer',
         pp.NumericalImputer(variables=NUMERICAL_VARS_WITH_NA)),
        ('temporal_variable_imputer',
         pp.TemporalVariableImputer(variables = TEMPORAL_VARS)),
        ('top_label_encoder',
         pp.TopLabelCategoricalEncoder(tol = 7, variables = HIGH_CARDINALITY_VARS)),
        ('categorical_encoder',
         pp.CategoricalEncoder(variables = CATEGORICAL_VARS)),
        #('drop_features',
        	#pp.DropUnecessaryFeatures(variables = DROP_VARS)),
        ('scaler', MinMaxScaler()),
        ('classification_model',
         RandomForestClassifier(n_estimators = 10, max_depth = 5, max_leaf_nodes = 45, random_state=0))
    ]
)

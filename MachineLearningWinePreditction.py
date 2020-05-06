%pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
train_df = sqlContext.read.format("com.databricks.spark.csv").format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "true").load("s3n://hadoop-data-files/training/TrainingDataset.csv")


from pyspark.ml.feature import VectorAssembler
vectorAssembler =VectorAssembler(inputCols=['fixed acidity',
                                     'volatile acidity',
                                     'citric acid',
                                     'residual sugar',
                                     'chlorides',
                                     'free sulfur dioxide',
                                     'total sulfur dioxide',
                                     'density',
                                     'pH',
                                     'sulphates'], outputCol='features')

vtrain_df = vectorAssembler.transform(train_df)
vtrain_df = vtrain_df.select(['features', 'alcohol'])


from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='alcohol', maxIter=10, regParam=0.3, elasticNetParam=0.8)

lr_model = lr.fit(vtrain_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
train_df.describe().show()

prediction_df = sqlContext.read.format("com.databricks.spark.csv").format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "true").load("s3n://hadoop-data-files/training/ValidationDataset.csv")
vpredict_df = vectorAssembler.transform(prediction_df)
vpredict_df = vpredict_df.select(['features'])

prediction_final_df = lr_model.transform(vpredict_df)
prediction_final_df.show()

prediction_final_df.describe().show()

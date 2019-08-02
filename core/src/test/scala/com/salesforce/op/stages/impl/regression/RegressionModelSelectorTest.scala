/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.evaluators._
import com.salesforce.op.features.types._
import com.salesforce.op.features.{Feature, FeatureBuilder}
import com.salesforce.op.stages.impl.CompareParamGrid
import com.salesforce.op.stages.impl.regression.{RegressionModelsToTry => RMT}
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.EstimatorType
import com.salesforce.op.stages.impl.selector.ModelSelectorSummary
import com.salesforce.op.stages.impl.tuning.BestEstimator
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import ml.dmlc.xgboost4j.scala.spark.OpXGBoostQuietLogging
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class RegressionModelSelectorTest extends FlatSpec with TestSparkContext
  with CompareParamGrid with OpXGBoostQuietLogging {
  val seed = 1234L
  val stageNames = "label_prediction"

  import spark.implicits._

  val rawData: Seq[(Double, Vector)] = List.range(0, 100, 1).map(i =>
    (i.toDouble, Vectors.dense(2 * i, 4 * i)))

  val data = sc.parallelize(rawData).toDF("label", "features")

  val (label, Array(features: Feature[OPVector]@unchecked)) = FeatureBuilder.fromDataFrame[RealNN](
    data, response = "label", nonNullable = Set("features")
  )

  val lr = new OpLinearRegression()
  val lrParams = new ParamGridBuilder()
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lr.maxIter, Array(10, 100))
    .addGrid(lr.regParam, Array(0.0))
    .addGrid(lr.solver, Array("l-bfgs"))
    .build()

  val rf = new OpRandomForestRegressor()
  val rfParams = new ParamGridBuilder()
    .addGrid(rf.maxDepth, Array(2, 10))
    .addGrid(rf.minInstancesPerNode, Array(1))
    .addGrid(rf.numTrees, Array(10))
    .build()

  val models = Seq(lr -> lrParams, rf -> rfParams)


  Spec(RegressionModelSelector.getClass) should "be properly set" in {
    val modelSelector = RegressionModelSelector().setInput(label.asInstanceOf[Feature[RealNN]], features)

    val inputNames = modelSelector.getInputFeatures().map(_.name)
    inputNames should have length 2
    inputNames shouldBe Array(label.name, features.name)
    modelSelector.getOutput().name shouldBe modelSelector.getOutputFeatureName
    the[IllegalArgumentException] thrownBy {
      modelSelector.setInput(label, features.copy(isResponse = true))
    } should have message "The feature vector should not contain any response features."
  }

  it should "properly select models to try" in {
    val modelSelector = RegressionModelSelector
      .withCrossValidation(
        modelTypesToUse = Seq(RMT.OpLinearRegression, RMT.OpRandomForestRegressor, RMT.OpXGBoostRegressor)
      )

    modelSelector.models.size shouldBe 3
    modelSelector.models.exists(_._1.getClass.getSimpleName == RMT.OpLinearRegression.entryName) shouldBe true
    modelSelector.models.exists(_._1.getClass.getSimpleName == RMT.OpRandomForestRegressor.entryName) shouldBe true
    modelSelector.models.exists(_._1.getClass.getSimpleName == RMT.OpGBTRegressor.entryName) shouldBe false
    modelSelector.models.exists(_._1.getClass.getSimpleName == RMT.OpXGBoostRegressor.entryName) shouldBe true
  }

  it should "set the data splitting params correctly" in {
    val modelSelector = RegressionModelSelector()
    modelSelector.splitter.get.setReserveTestFraction(0.1).setSeed(11L)
    modelSelector.splitter.get.getSeed shouldBe 11L
    modelSelector.splitter.get.getReserveTestFraction shouldBe 0.1
  }

  it should "split into training and test" in {

    implicit val vectorEncoder: org.apache.spark.sql.Encoder[Vector] = ExpressionEncoder()
    implicit val e1 = Encoders.tuple(Encoders.scalaDouble, vectorEncoder)
    val modelSelector = RegressionModelSelector.withCrossValidation(seed = seed)
    val (train, test) = modelSelector.setInput(label, features)
      .splitter.get.setReserveTestFraction(0.2).split(data.as[(Double, Vector)])

    val trainCount = train.count()
    val testCount = test.count()
    val totalCount = rawData.length

    assert(math.abs(testCount - 0.2 * totalCount) <= 15)
    assert(math.abs(trainCount - 0.8 * totalCount) <= 15)

    trainCount + testCount shouldBe totalCount
  }

  ignore should "fit and predict all models on by default" in {

    val testEstimator = RegressionModelSelector
      .withCrossValidation(
        numFolds = 4,
        validationMetric = Evaluators.Regression.mse(),
        seed = 10L
      )
      .setInput(label, features)


    val model = testEstimator.fit(data)
    model.evaluateModel(data)
    val pred = model.getOutput()

    // evaluation metrics from train set should be in metadata
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
    RegressionEvalMetrics.values.foreach(metric =>
      assert(metaData.trainEvaluation.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    metaData.validationResults.length shouldEqual 52

    // evaluation metrics from train set should be in metadata
    val metaDataHoldOut = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata()).holdoutEvaluation
    RegressionEvalMetrics.values.foreach(metric =>
      assert(metaDataHoldOut.get.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    val transformedData = model.transform(data)
    val justScores = transformedData.collect(pred)

    assertScores(justScores, transformedData.collect(label))
  }

  it should "fit and predict for default models" in {
    // Take one random model type each time
    val defaultModels = RegressionModelSelector.Defaults.modelTypesToUse
    val modelToTry = defaultModels(scala.util.Random.nextInt(defaultModels.size))

    val testEstimator = RegressionModelSelector
      .withCrossValidation(
        numFolds = 4,
        validationMetric = Evaluators.Regression.mse(),
        seed = 10L,
        modelTypesToUse = Seq(modelToTry)
      )
      .setInput(label, features)


    val model = testEstimator.fit(data)
    model.evaluateModel(data)
    val pred = model.getOutput()

    // evaluation metrics from train set should be in metadata
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
    RegressionEvalMetrics.values.foreach(metric =>
      assert(metaData.trainEvaluation.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    // evaluation metrics from train set should be in metadata
    val metaDataHoldOut = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata()).holdoutEvaluation
    RegressionEvalMetrics.values.foreach(metric =>
      assert(metaDataHoldOut.get.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    val transformedData = model.transform(data)
    val justScores = transformedData.collect(pred)

    justScores.length shouldEqual transformedData.count()
  }

  it should "fit and predict with a train validation split even if there is no split between training and test" in {
    val testEstimator =
      RegressionModelSelector
        .withTrainValidationSplit(
          dataSplitter = None,
          trainRatio = 0.8,
          validationMetric = Evaluators.Regression.r2(),
          seed = 10L,
          modelsAndParameters = models
        )
        .setInput(label, features)
    val pred = testEstimator.getOutput()
    val transformedData = testEstimator.fit(data).transform(data)
    val justScores = transformedData.collect(pred)

    assertScores(justScores, transformedData.collect(label))
  }

  it should "fit and predict correctly with MeanAbsoluteError used" in {
    val testEstimator =
      RegressionModelSelector
        .withCrossValidation(numFolds = 4, validationMetric = Evaluators.Regression.mae(), seed = 11L,
          modelsAndParameters = models)
        .setInput(label, features)
    val pred = testEstimator.getOutput()
    val transformedData = testEstimator.fit(data).transform(data)
    val justScores = transformedData.collect(pred)

    assertScores(justScores, transformedData.collect(label))
  }

  it should "fit and predict correctly with a custom metric used" in {

    val medianAbsoluteError = Evaluators.Regression.custom(
      metricName = "median absolute error",
      isLargerBetter = false,
      evaluateFn = ds => {
        val medAE = ds.map { case (lbl, prediction) => math.abs(prediction - lbl) }
        val median = medAE.stat.approxQuantile(medAE.columns.head, Array(0.5), 0.25)
        median.head
      }
    )
    val testEstimator =
      RegressionModelSelector
        .withCrossValidation(numFolds = 4, validationMetric = medianAbsoluteError, seed = 11L,
          modelsAndParameters = models)
        .setInput(label, features)
    val pred = testEstimator.getOutput()
    val transformedData = testEstimator.fit(data).transform(data)
    val justScores = transformedData.collect(pred)

    assertScores(justScores, transformedData.collect(label))
  }

  it should "fit correctly with a custom metric used as holdOut evaluator" in {

    val medianAbsoluteError = Evaluators.Regression.custom(
      metricName = "median absolute error",
      isLargerBetter = false,
      evaluateFn = ds => {
        val medAE = ds.map { case (lbl, prediction) => math.abs(prediction - lbl) }
        val median = medAE.stat.approxQuantile(medAE.columns.head, Array(0.5), 0.25)
        median.head
      }
    )
    val testEstimator =
      RegressionModelSelector
        .withCrossValidation(numFolds = 4, validationMetric = medianAbsoluteError,
          trainTestEvaluators = Seq(medianAbsoluteError), seed = 11L, modelsAndParameters = models)
        .setInput(label, features)
    val model = testEstimator.fit(data)

    // checking trainingEval & holdOutEval metrics
    model.evaluateModel(data)
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
    val trainMetaData = metaData.trainEvaluation
    val holdOutMetaData = metaData.holdoutEvaluation.get

    testEstimator.evaluators.foreach {
      case evaluator: OpRegressionEvaluator => {
        RegressionEvalMetrics.values.foreach(metric =>
          Seq(trainMetaData, holdOutMetaData).foreach(
            metadata => assert(metadata.toJson(false).contains(s"${metric.entryName}"),
              s"Metric ${metric.entryName} is not present in metadata: " + metadata)
          )
        )
      }
      case evaluator: OpRegressionEvaluatorBase[_] => {
        Seq(trainMetaData, holdOutMetaData).foreach(
          metadata =>
            assert(metadata.toJson(false).contains(s"${evaluator.name.humanFriendlyName}"),
              s"Single Metric evaluator ${evaluator.name} is not present in metadata: " + metadata)
        )
      }
    }
  }

  it should "fit and predict a model specified in the var bestEstimator" in {
    val modelSelector = RegressionModelSelector().setInput(label, features)
    val myParam = true
    val myEstimatorName = "myEstimatorIsAwesome"
    val myEstimator = new OpGBTRegressor().setCacheNodeIds(myParam).setInput(label, features)

    val bestEstimator = new BestEstimator(myEstimatorName, myEstimator.asInstanceOf[EstimatorType], Seq.empty)
    modelSelector.bestEstimator = Option(bestEstimator)
    val fitted = modelSelector.fit(data)

    fitted.modelStageIn.parent.extractParamMap().toSeq
      .collect{ case p: ParamPair[_] if p.param.name == "cacheNodeIds" => p.value }.head shouldBe myParam

    val meta = ModelSelectorSummary.fromMetadata(fitted.getMetadata().getSummaryMetadata())
    meta.bestModelName shouldBe myEstimatorName
  }

  private def assertScores(scores: Array[Prediction], labels: Array[RealNN]) = {
    val res = scores.zip(labels)
      .map { case (score: Prediction, label: RealNN) => math.abs(score.prediction - label.v.get) }.sum
    assert(res <= scores.length, "prediction failed")
  }
}

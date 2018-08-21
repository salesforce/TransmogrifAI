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

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.evaluators._
import com.salesforce.op.features.types._
import com.salesforce.op.features.{Feature, FeatureBuilder}
import com.salesforce.op.stages.impl.CompareParamGrid
import com.salesforce.op.stages.impl.classification.FunctionalityForClassificationTests._
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelsToTry => BMT}
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.EstimatorType
import com.salesforce.op.stages.impl.selector.{ModelEvaluation, ModelSelectorNames, ModelSelectorSummary}
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory

@RunWith(classOf[JUnitRunner])
class BinaryClassificationModelSelectorTest extends FlatSpec with TestSparkContext with CompareParamGrid {

  val log = LoggerFactory.getLogger(this.getClass)

  val (seed, smallCount, bigCount) = (1234L, 20, 80)

  import spark.implicits._

  // Generate positive observations following a distribution ~ N((0.0, 0.0, 0.0), I_3)
  val positiveData =
    normalVectorRDD(spark.sparkContext, bigCount, 3, seed = seed)
      .map(v => 1.0 -> Vectors.dense(v.toArray))

  // Generate negative observations following a distribution ~ N((10.0, 10.0, 10.0), I_3)
  val negativeData =
    normalVectorRDD(spark.sparkContext, smallCount, 3, seed = seed)
      .map(v => 0.0 -> Vectors.dense(v.toArray.map(_ + 10.0)))

  val stageNames = Array("label_prediction", "label_rawPrediction", "label_probability")

  val data = positiveData.union(negativeData).toDF("label", "features")

  val (label, Array(features: Feature[OPVector]@unchecked)) = FeatureBuilder.fromDataFrame[RealNN](
    data, response = "label", nonNullable = Set("features")
  )

  val lr = new OpLogisticRegression()
  val lrParams = new ParamGridBuilder()
    .addGrid(lr.elasticNetParam, Array(1.0))
    .addGrid(lr.maxIter, Array(10))
    .addGrid(lr.regParam, Array(1000000.0, 0.0))
    .build()

  val rf = new OpRandomForestClassifier()
  val rfParams = new ParamGridBuilder()
    .addGrid(rf.maxDepth, Array(0))
    .addGrid(rf.minInstancesPerNode, Array(1))
    .addGrid(rf.numTrees, Array(10))
    .addGrid(rf.minInfoGain, Array(10000.0))
    .build()

  val models = Seq(lr -> lrParams, rf -> rfParams)

  Spec(BinaryClassificationModelSelector.getClass) should  "properly select models to try" in {
    val modelSelector = BinaryClassificationModelSelector
      .withCrossValidation(modelTypesToUse = Seq(BMT.OpLogisticRegression, BMT.OpRandomForestClassifier))
      .setInput(label.asInstanceOf[Feature[RealNN]], features)

    modelSelector.models.size shouldBe 2
    modelSelector.models.exists(_._1.getClass.getSimpleName == BMT.OpLogisticRegression.entryName) shouldBe true
    modelSelector.models.exists(_._1.getClass.getSimpleName == BMT.OpRandomForestClassifier.entryName) shouldBe true
    modelSelector.models.exists(_._1.getClass.getSimpleName == BMT.OpNaiveBayes.entryName) shouldBe false
  }

  it should "split into training and test even if the balancing is not desired" in {
    implicit val vectorEncoder: org.apache.spark.sql.Encoder[Vector] = ExpressionEncoder()
    implicit val e1 = Encoders.tuple(Encoders.scalaDouble, vectorEncoder)

    val testFraction = 0.2

    val (train, test) = DataSplitter(reserveTestFraction = testFraction)
      .split(data.withColumn(ModelSelectorNames.idColName, monotonically_increasing_id())
        .as[(Double, Vector, Double)])

    val trainCount = train.count()
    val testCount = test.count()
    val totalCount = smallCount + bigCount

    assert(math.abs(testCount - testFraction * totalCount) <= 20)
    assert(math.abs(trainCount - (1.0 - testFraction) * totalCount) <= 20)

    trainCount + testCount shouldBe totalCount
  }

  ignore should "fit and predict all models on by default" in {

    val testEstimator =
      BinaryClassificationModelSelector
        .withCrossValidation(
          Option(DataBalancer(sampleFraction = 0.5, seed = 11L)),
          numFolds = 4,
          validationMetric = Evaluators.BinaryClassification.error(),
          seed = 42L
        )
        .setInput(label, features)

    val model = testEstimator.fit(data)
    log.info(model.getMetadata().toString)

    // Evaluation from train data should be there
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaData.trainEvaluation.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )
    metaData.validationResults.length shouldEqual 48

    // evaluation metrics from test set should be in metadata after eval run
    model.evaluateModel(data)
    val metaDataHoldOut = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
      .holdoutEvaluation
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaDataHoldOut.get.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    val transformedData = model.transform(data)
    val pred = model.getOutput()
    val justScores = transformedData.collect(pred).map(_.prediction)
    justScores shouldEqual data.collect(label).map(_.v.get)
  }

  it should "fit and predict for default models" in {

    val modelToTry = BinaryClassificationModelSelector.modelNames(scala.util.Random.nextInt(4))

    val testEstimator =
      BinaryClassificationModelSelector
        .withCrossValidation(
          Option(DataBalancer(sampleFraction = 0.5, seed = 11L)),
          numFolds = 4,
          validationMetric = Evaluators.BinaryClassification.error(),
          seed = 42,
          modelTypesToUse = Seq(modelToTry)
        )
        .setInput(label, features)

    val model = testEstimator.fit(data)
    log.info(model.getMetadata().toString)

    // Evaluation from train data should be there
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaData.trainEvaluation.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    // evaluation metrics from test set should be in metadata after eval run
    model.evaluateModel(data)
    val metaDataHoldOut = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
      .holdoutEvaluation
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaDataHoldOut.get.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    val transformedData = model.transform(data)
    val pred = model.getOutput()
    val justScores = transformedData.collect(pred).map(_.prediction)
    justScores.length shouldEqual transformedData.count()
  }

  it should "fit and predict even if there is no balancing" in {

    val testEstimator =
      BinaryClassificationModelSelector
        .withCrossValidation(
          Option(DataSplitter(reserveTestFraction = 0.2)),
          numFolds = 4,
          validationMetric = Evaluators.BinaryClassification.auPR(),
          seed = 10L,
          modelsAndParameters = models
        )
        .setInput(label, features)

    val model = testEstimator.fit(data)

    log.info(model.getMetadata().toString)

    val sparkStage = model.modelStageIn
    sparkStage.isInstanceOf[OpLogisticRegressionModel] shouldBe true
    sparkStage.parent.extractParamMap()(sparkStage.parent.getParam("maxIter")) shouldBe 10
    sparkStage.parent.extractParamMap()(sparkStage.parent.getParam("regParam")) shouldBe 0.0

    // evaluation metrics from test set should be in metadata
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaData.trainEvaluation.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    // evaluation metrics from test set should be in metadata after eval run
    model.evaluateModel(data)
    val metaDataHoldOut = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata()).holdoutEvaluation
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaDataHoldOut.get.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    val transformedData = model.transform(data)
    val pred = model.getOutput()
    val justScores = transformedData.collect(pred).map(_.prediction)
    justScores shouldEqual data.collect(label).map(_.v.get)
  }

  it should "fit and predict with a train validation split, " +
    "even if there is no split between training and test + custom evaluator" in {

    val crossEntropy = Evaluators.BinaryClassification.custom(
      metricName = "cross entropy",
      isLargerBetter = false,
      evaluateFn = crossEntropyFun
    )

    val testEstimator =
      BinaryClassificationModelSelector
        .withTrainValidationSplit(Option(DataBalancer(sampleFraction = 0.2, seed = 11L, reserveTestFraction = 0.0)),
          trainRatio = 0.8, validationMetric = crossEntropy, seed = 10L, modelsAndParameters = models)
        .setInput(label, features)

    val model = testEstimator.fit(data)

    log.info(model.getMetadata().toString)

    val transformedData = model.transform(data)
    val pred = testEstimator.getOutput()
    val justScores = transformedData.collect(pred).map(_.prediction)
    justScores shouldEqual data.collect(label).map(_.v.get)
  }

  it should "fit and predict with a train validation split, even if there is no split and balancing" in {

    val testEstimator =
      BinaryClassificationModelSelector
        .withTrainValidationSplit(None,
          trainRatio = 0.8, validationMetric = Evaluators.BinaryClassification.precision(), seed = 10L,
          modelsAndParameters = models)
        .setInput(label, features)

    val model = testEstimator.fit(data)

    log.info(model.getMetadata().toString)

    val transformedData = model.transform(data)
    val pred = testEstimator.getOutput()
    val justScores = transformedData.collect(pred).map(_.prediction)

    justScores shouldEqual transformedData.collect(label).map(_.v.get)
  }

  it should "fit and predict with a cross validation and compute correct metrics from evaluators" in {

    val crossEntropy = Evaluators.BinaryClassification.custom(
      metricName = "cross entropy",
      isLargerBetter = false,
      evaluateFn = crossEntropyFun
    )

    val testEstimator =
      BinaryClassificationModelSelector
        .withCrossValidation(
          Option(DataSplitter(reserveTestFraction = 0.2)),
          numFolds = 4,
          validationMetric = Evaluators.BinaryClassification.recall(),
          trainTestEvaluators = Seq(crossEntropy),
          seed = 10L,
          modelsAndParameters = models
        )
        .setInput(label, features)

    val model = testEstimator.fit(data)
    model.evaluateModel(data)

    // checking the holdOut Evaluators
    assert(testEstimator.evaluators.contains(crossEntropy), "Cross entropy evaluator not present in estimator")

    // checking trainingEval & holdOutEval metrics
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata())
    val trainMetaData = metaData.trainEvaluation
    val holdOutMetaData = metaData.holdoutEvaluation.get

    testEstimator.evaluators.foreach {
      case evaluator: OpBinaryClassificationEvaluator => {
        BinaryClassEvalMetrics.values.foreach(metric =>
          Seq(trainMetaData, holdOutMetaData).foreach(
            metadata => assert(metadata.toJson(false).contains(s"${metric.entryName}"),
              s"Metric ${metric.entryName} is not present in metadata: " + metadata)
          )
        )
      }
      case evaluator: OpBinaryClassificationEvaluatorBase[_] => {
        Seq(trainMetaData, holdOutMetaData).foreach(metadata =>
          assert(metadata.toJson(false).contains(s"${evaluator.name.humanFriendlyName}"),
            s"Single Metric evaluator ${evaluator.name} is not present in metadata: " + metadata)
        )
      }
    }
  }

  it should "fit and predict a model specified in the var bestEstimator" in {
    val modelSelector = BinaryClassificationModelSelector().setInput(label, features)
    val myParam = 42
    val myMetaName = "myMeta"
    val myMetaValue = 348954389534875.432423
    val myMetadata = ModelEvaluation(myMetaName, myMetaName, myMetaName, SingleMetric(myMetaName, myMetaValue),
      Map.empty)
    val myEstimatorName = "myEstimator"
    val myEstimator = new OpLogisticRegression().setMaxIter(myParam).setInput(label, features)

    val bestEstimator = new BestEstimator(myEstimatorName, myEstimator.asInstanceOf[EstimatorType], Seq(myMetadata))
    modelSelector.bestEstimator = Option(bestEstimator)
    val fitted = modelSelector.fit(data)

    fitted.modelStageIn.parent.extractParamMap().toSeq
      .collect{ case p: ParamPair[_] if p.param.name == "maxIter" => p.value }.head shouldBe myParam

    val meta = ModelSelectorSummary.fromMetadata(fitted.getMetadata().getSummaryMetadata())
    meta.validationResults.head shouldBe myMetadata
  }

}

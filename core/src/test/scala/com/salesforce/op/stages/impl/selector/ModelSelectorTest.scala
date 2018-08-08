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

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.UID
import com.salesforce.op.evaluators._
import com.salesforce.op.features.types._
import com.salesforce.op.features.{Feature, FeatureBuilder}
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.CompareParamGrid
import com.salesforce.op.stages.impl.classification.{OpLogisticRegression, OpLogisticRegressionModel, OpRandomForestClassifier}
import com.salesforce.op.stages.impl.regression.{OpLinearRegression, OpLinearRegressionModel, OpRandomForestRegressor}
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpPredictorWrapperModel}
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.classification.{LogisticRegression => SparkLR}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory

@RunWith(classOf[JUnitRunner])
class ModelSelectorTest extends FlatSpec with TestSparkContext with CompareParamGrid {

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

  val data = positiveData.union(negativeData).toDF("label", "features")

  val (label, Array(features: Feature[OPVector]@unchecked)) = FeatureBuilder.fromDataFrame[RealNN](
    data, response = "label", nonNullable = Set("features")
  )

  private val lr = new OpLogisticRegression()
  private val lrParams = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(0.1, 100))
    .addGrid(lr.elasticNetParam, Array(0, 0.5)).build()

  private val rf = new OpRandomForestClassifier()
  private val rfParams = new ParamGridBuilder()
    .addGrid(rf.numTrees, Array(2, 4))
    .addGrid(rf.minInfoGain, Array(100.0, 10.0)).build()

  private val linR = new OpLinearRegression()
  private val linRParams = new ParamGridBuilder()
    .addGrid(linR.regParam, Array(0.1, 100))
    .addGrid(linR.maxIter, Array(10, 20)).build()

  private val rfR = new OpRandomForestRegressor()
  private val rfRParams = new ParamGridBuilder()
    .addGrid(rfR.numTrees, Array(2, 4))
    .addGrid(rfR.minInfoGain, Array(100.0, 10.0)).build()


  Spec[ModelSelector[_, _]] should "fit and predict classifiers" in {

    val validatorCV = new OpCrossValidation[OpPredictorWrapperModel[_], OpPredictorWrapper[_, _]](
      numFolds = 3, seed = seed, Evaluators.BinaryClassification.auPR(), stratify = false, parallelism = 1
    )

    val testEstimator = new ModelSelector(
      validator = validatorCV,
      splitter = Option(DataBalancer(sampleFraction = 0.5, seed = 11L)),
      models = Seq(lr -> lrParams, rf -> rfParams),
      evaluators = Seq(new OpBinaryClassificationEvaluator)
    ).setInput(label, features)

    val model = testEstimator.fit(data)
    model.modelStageIn.isInstanceOf[OpLogisticRegressionModel] shouldBe true

    val bestEstimator = model.modelStageIn.parent
    bestEstimator.extractParamMap()(bestEstimator.getParam("regParam")) shouldBe 0.1

    log.info(model.getMetadata().toString)
    // Evaluation from train data should be there
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata()).trainEvaluation
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaData.toJson(false).contains(s"${metric.entryName}"),
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
    justScores shouldEqual data.collect(label).map(_.value.get)
  }

  it should "fit and predict regressors" in {

    val validatorTS = new OpTrainValidationSplit[OpPredictorWrapperModel[_], OpPredictorWrapper[_, _]](
      trainRatio = 0.8, seed = seed, Evaluators.Regression.r2(), stratify = false, parallelism = 1
    )

    val testEstimator = new ModelSelector(
      validator = validatorTS,
      splitter = Option(DataBalancer(sampleFraction = 0.5, seed = 11L)),
      models = Seq(linR -> linRParams, rfR -> rfRParams),
      evaluators = Seq(new OpRegressionEvaluator())
    ).setInput(label, features)

    val model = testEstimator.fit(data)
    model.modelStageIn.isInstanceOf[OpLinearRegressionModel] shouldBe true

    val bestEstimator = model.modelStageIn.parent
    bestEstimator.extractParamMap()(bestEstimator.getParam("regParam")) shouldBe 0.1

    log.info(model.getMetadata().toString)
    // Evaluation from train data should be there
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata()).trainEvaluation
    RegressionEvalMetrics.values.foreach(metric =>
      assert(metaData.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    // evaluation metrics from test set should be in metadata after eval run
    model.evaluateModel(data)
    val metaDataHoldOut = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata()).holdoutEvaluation
    RegressionEvalMetrics.values.foreach(metric =>
      assert(metaDataHoldOut.get.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    val transformedData = model.transform(data)
    val pred = model.getOutput()
    val justScores = transformedData.collect(pred).map(p => math.round(p.prediction))
    justScores shouldEqual data.collect(label).map(_.value.get)

  }

  it should "take estimators which extend OpPipelineStage2[RealNN, OPVector, Prediction]" in {
    val test = new TestEstimator()
    val testParams = Array(new ParamMap())

    val lrParams2 = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(100.0))
      .addGrid(lr.elasticNetParam, Array(0, 0.5)).build()

    val validatorCV = new OpCrossValidation[ModelSelectorBaseNames.ModelType, ModelSelectorBaseNames.EstimatorType](
      numFolds = 3, seed = seed, Evaluators.BinaryClassification.auPR(),
      stratify = false, parallelism = 1)

    val testEstimator = new ModelSelector(
      validator = validatorCV,
      splitter = Option(DataBalancer(sampleFraction = 0.5, seed = 11L)),
      models = Seq(lr -> lrParams2, test -> testParams),
      evaluators = Seq(new OpBinaryClassificationEvaluator)
    ).setInput(label, features)

    val model = testEstimator.fit(data)

    log.info(model.getMetadata().toString)
    // Evaluation from train data should be there
    val metaData = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata()).trainEvaluation
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaData.toJson(false).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    // evaluation metrics from test set should be in metadata after eval run
    model.evaluateModel(data)
    val metaDataHoldOut = ModelSelectorSummary.fromMetadata(model.getMetadata().getSummaryMetadata()).holdoutEvaluation
    BinaryClassEvalMetrics.values.foreach(metric =>
      assert(metaDataHoldOut.get.toJson(true).contains(s"${metric.entryName}"),
        s"Metric ${metric.entryName} is not present in metadata: " + metaData)
    )

    val transformedData = model.transform(data)
    val pred = model.getOutput()
    val justScores = transformedData.collect(pred)
    justScores.size shouldEqual data.count()

  }

}

class TestEstimator extends BinaryEstimator[RealNN, OPVector, Prediction]("test", UID[TestEstimator]) {
  override def fitFn(dataset: Dataset[(Option[Double], Vector)]): TestModel = new TestModel(uid)
}

class TestModel(uid: String) extends BinaryModel[RealNN, OPVector, Prediction]("test", uid){
  override def transformFn: (RealNN, OPVector) => Prediction = (l: RealNN, f: OPVector) => {
    val pred = l.value.get
    val raw = Array(pred, 1 - pred)
    Prediction(pred, raw, raw)
  }
}

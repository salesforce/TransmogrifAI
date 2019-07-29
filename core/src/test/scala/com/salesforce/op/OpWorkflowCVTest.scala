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

package com.salesforce.op

import java.nio.file.Paths

import com.salesforce.app.schema.PassengerDataAll
import com.salesforce.op.evaluators.{OpMultiClassificationEvaluator, _}
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.readers._
import com.salesforce.op.stages.base.binary.BinaryTransformer
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.preparators.SanityChecker
import com.salesforce.op.stages.impl.regression._
import com.salesforce.op.stages.impl.selector.ModelSelector
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.spark.RichParamMap._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory

@RunWith(classOf[JUnitRunner])
class OpWorkflowCVTest extends FlatSpec with PassengerSparkFixtureTest {

  val log = LoggerFactory.getLogger(this.getClass)

  trait PassenserCSVforCV {
    val simplePassengerForCV = DataReaders.Simple.csv[PassengerDataAll](
      path = Some(Paths.get(testDataDir, "PassengerDataAll.csv").toString),
      schema = PassengerDataAll.getClassSchema.toString,
      key = _.getPassengerId.toString
    )
    val age = FeatureBuilder.Real[PassengerDataAll].extract(_.getAge.toReal).asPredictor
    val sex = FeatureBuilder.PickList[PassengerDataAll].extract(_.getSex.toPickList).asPredictor
    val fair = FeatureBuilder.Real[PassengerDataAll].extract(p => Option(p.getFare).map(_.toDouble).toReal).asPredictor
    val pClass = FeatureBuilder.PickList[PassengerDataAll].extract(_.getPclass.toString.toPickList).asPredictor
    val cabin = FeatureBuilder.PickList[PassengerDataAll].extract(_.getCabin.toPickList).asPredictor
    val survived = FeatureBuilder.Binary[PassengerDataAll].extract(p => p.getSurvived.intValue.toBinary).asResponse
    val survivedPred = FeatureBuilder.Binary[PassengerDataAll].extract(p => p.getSurvived.intValue.toBinary).asPredictor
    val survivedNum = survived.occurs()
  }

  Spec[OpWorkflow] should
    "return a binary classification model that runs cv at the workflow level" in new PassenserCSVforCV {
    val fv = Seq(age, sex, fair, pClass, cabin).transmogrify()
    val checked = survivedNum.sanityCheck(fv)

    val lr = new OpLogisticRegression()
    val lrParams = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.01, 0.5))
      .addGrid(lr.regParam, Array(10000.0))
      .build()

    val rf = new OpRandomForestClassifier()
    val rfParams = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(10))
      .build()

    val models = Seq(lr -> lrParams, rf -> rfParams)

    val pred1 = BinaryClassificationModelSelector.withCrossValidation(
      splitter = Option(DataBalancer(sampleFraction = 0.01, reserveTestFraction = 0.2, seed = 0L)),
      numFolds = 2,
      validationMetric = Evaluators.BinaryClassification.auPR(),
      trainTestEvaluators = Seq(new OpBinaryClassificationEvaluator),
      seed = 10L,
      modelsAndParameters = models
    )
      .setInput(survivedNum, checked)
      .getOutput()

    val wf1 = new OpWorkflow().withWorkflowCV.setResultFeatures(pred1)
    wf1.isWorkflowCV shouldBe true
    val model1 = wf1.setReader(simplePassengerForCV).train()
    val data1 = model1.score(keepRawFeatures = false, keepIntermediateFeatures = false)

    val pred2 = BinaryClassificationModelSelector.withCrossValidation(
      splitter = Option(DataBalancer(sampleFraction = 0.01, reserveTestFraction = 0.2, seed = 0L)),
      numFolds = 2,
      validationMetric = Evaluators.BinaryClassification.auPR(),
      trainTestEvaluators = Seq(new OpBinaryClassificationEvaluator),
      seed = 10L,
      modelsAndParameters = models
    )
      .setInput(survivedNum, checked)
      .getOutput()


    val wf2 = new OpWorkflow().setResultFeatures(pred2)
    wf2.isWorkflowCV shouldBe false
    val model2 = wf2.setReader(simplePassengerForCV).train()
    val data2 = model2.score(keepRawFeatures = false, keepIntermediateFeatures = false)

    compare(data1, data2, pred1, pred2)

    val summary = model1.summary()
    summary should include (classOf[SanityChecker].getSimpleName)
    summary should include (""""HoldoutEvaluation" : [ "com.salesforce.op.evaluators.MultiMetrics"""")
    summary should include ("TrainEvaluation")
  }

  it should "return a multi classification model that runs ts at the workflow level" in new PassenserCSVforCV {
    val fv = Seq(age, sex, fair, pClass, cabin).transmogrify()
    val checked = survivedNum.sanityCheck(fv)

    val lr = new OpLogisticRegression()
    val lrParams = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1))
      .build()

    val dt = new OpDecisionTreeClassifier()
    val dtParams = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(5, 10))
      .addGrid(dt.minInfoGain, Array(100000.0))
      .build()

    val models = Seq(lr -> lrParams, dt -> dtParams)

    val pred1 = MultiClassificationModelSelector.withTrainValidationSplit(
      splitter = Option(DataCutter(reserveTestFraction = 0.2, seed = 0L)),
      validationMetric = Evaluators.MultiClassification.error(),
      trainTestEvaluators = Seq(new OpMultiClassificationEvaluator()),
      seed = 10L,
      modelsAndParameters = models
    ).setInput(survivedNum, checked)
      .getOutput()


    val wf1 = new OpWorkflow().withWorkflowCV.setResultFeatures(pred1)
    wf1.isWorkflowCV shouldBe true
    val model1 = wf1.setReader(simplePassengerForCV).train()
    val data1 = model1.score(keepRawFeatures = false, keepIntermediateFeatures = false)


    val pred2 = MultiClassificationModelSelector.withTrainValidationSplit(
      splitter = Option(DataCutter(reserveTestFraction = 0.2, seed = 0L)),
      validationMetric = Evaluators.MultiClassification.error(),
      trainTestEvaluators = Seq(new OpMultiClassificationEvaluator()),
      seed = 10L,
      modelsAndParameters = models
    ).setInput(survivedNum, checked)
      .getOutput()

    val wf2 = new OpWorkflow().setResultFeatures(pred2)
    wf2.isWorkflowCV shouldBe false
    val model2 = wf2.setReader(simplePassengerForCV).train()
    val data2 = model2.score(keepRawFeatures = false, keepIntermediateFeatures = false)

    compare(data1, data2, pred1, pred2)

    val summary = model1.summary()
    log.info(summary)
    summary should include (classOf[SanityChecker].getSimpleName)
    summary should include (""""HoldoutEvaluation" : [ "com.salesforce.op.evaluators.MultiMetrics"""")
    summary should include ("TrainEvaluation")
  }

  it should "return a regression model that runs cv at the workflow level" in new PassenserCSVforCV {
    val fv = Seq(sex, fair, pClass, cabin, age).transmogrify()
    val checked = survivedNum.sanityCheck(fv)

    val lr = new OpLinearRegression()
    val lrParams = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01))
      .build()

    val rf = new OpRandomForestRegressor()
    val rfParams = new ParamGridBuilder()
      .addGrid(rf.minInfoGain, Array(100000.0))
      .build()

    val models = Seq(lr -> lrParams, rf -> rfParams)

    val pred1 = RegressionModelSelector.withCrossValidation(
      dataSplitter = None,
      validationMetric = Evaluators.Regression.r2(),
      trainTestEvaluators = Seq(new OpRegressionEvaluator()),
      parallelism = 4,
      modelsAndParameters = models
    ).setInput(survivedNum, checked)
      .getOutput()

    val wf1 = new OpWorkflow().withWorkflowCV.setResultFeatures(pred1)
    wf1.isWorkflowCV shouldBe true
    val model1 = wf1.setReader(simplePassengerForCV).train()
    val data1 = model1.score(keepRawFeatures = false, keepIntermediateFeatures = false)

    val pred2 = RegressionModelSelector.withCrossValidation(
      dataSplitter = None,
      validationMetric = Evaluators.Regression.r2(),
      trainTestEvaluators = Seq(new OpRegressionEvaluator()),
      parallelism = 4,
      modelsAndParameters = models
    ).setInput(survivedNum, checked)
      .getOutput()

    val wf2 = new OpWorkflow().setResultFeatures(pred2)
    wf2.isWorkflowCV shouldBe false
    val model2 = wf2.setReader(simplePassengerForCV).train()
    val data2 = model2.score(keepRawFeatures = false, keepIntermediateFeatures = false)

    compare(data1, data2, pred1, pred2)

    val summary = model1.summary()
    log.info(summary)
    summary should include (classOf[SanityChecker].getSimpleName)
    summary should include ("TrainEvaluation")
  }

  it should "return a regression model that runs ts at the workflow level" in new PassenserCSVforCV {
    val fv = Seq(sex, fair, pClass, cabin, age).transmogrify()
    val checked = survivedNum.sanityCheck(fv)

    val lr = new OpLinearRegression()
    val lrParams = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10000.0))
      .build()

    val gbt = new OpGBTRegressor()
    val gbtParams = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, Array(5))
      .build()

    val models = Seq(lr -> lrParams, gbt -> gbtParams)

    val pred1 = RegressionModelSelector.withTrainValidationSplit(
      dataSplitter = Option(DataSplitter(seed = 0L)),
      validationMetric = Evaluators.Regression.r2(),
      trainTestEvaluators = Seq(new OpRegressionEvaluator()),
      parallelism = 4,
      modelsAndParameters = models
    ).setInput(survivedNum, checked)
      .getOutput()

    val wf1 = new OpWorkflow().withWorkflowCV.setResultFeatures(pred1)
    wf1.isWorkflowCV shouldBe true
    val model1 = wf1.setReader(simplePassengerForCV).train()
    val data1 = model1.score(keepRawFeatures = false, keepIntermediateFeatures = false)

    val pred2 = RegressionModelSelector.withTrainValidationSplit(
      dataSplitter = Option(DataSplitter(seed = 0L)),
      validationMetric = Evaluators.Regression.r2(),
      trainTestEvaluators = Seq(new OpRegressionEvaluator()),
      parallelism = 4,
      modelsAndParameters = models
    ).setInput(survivedNum, checked)
      .getOutput()

    val wf2 = new OpWorkflow().setResultFeatures(pred2)
    wf2.isWorkflowCV shouldBe false
    val model2 = wf2.setReader(simplePassengerForCV).train()
    val data2 = model2.score(keepRawFeatures = false, keepIntermediateFeatures = false)

    compare(data1, data2, pred1, pred2)

    val summary = model1.summary()
    log.info(summary)
    summary should include (classOf[SanityChecker].getSimpleName)
    summary should include (""""HoldoutEvaluation" : [ "com.salesforce.op.evaluators.MultiMetrics"""")
    summary should include ("TrainEvaluation")
  }

  it should "avoid adding label leakage when feature engineering would introduce it" in new PassenserCSVforCV {
    val fairLeaker = fair.autoBucketize(survivedNum, trackNulls = false)
    val ageLeaker = age.autoBucketize(survivedNum, trackNulls = false)
    val fv = Seq(age, sex, ageLeaker, fairLeaker, pClass, cabin).transmogrify()

    val lr = new OpLogisticRegression()
    val lrParams = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.0, 0.001, 0.1))
      .build()

    val models = Seq(lr -> lrParams)

    val pred1 = BinaryClassificationModelSelector.withCrossValidation(
      splitter = Option(DataBalancer(sampleFraction = 0.01, reserveTestFraction = 0.2, seed = 0L)),
      numFolds = 2,
      validationMetric = Evaluators.BinaryClassification.auPR(),
      trainTestEvaluators = Seq(new OpBinaryClassificationEvaluator),
      parallelism = 4,
      modelsAndParameters = models
    ).setInput(survivedNum, fv)
      .getOutput()

    val wf1 = new OpWorkflow().withWorkflowCV.setResultFeatures(pred1)
    wf1.isWorkflowCV shouldBe true
    val model1 = wf1.setReader(simplePassengerForCV).train()
    val data1 = model1.score(keepRawFeatures = false, keepIntermediateFeatures = true)

    val pred2 = BinaryClassificationModelSelector.withCrossValidation(
      splitter = Option(DataBalancer(sampleFraction = 0.01, reserveTestFraction = 0.2, seed = 0L)),
      numFolds = 2,
      validationMetric = Evaluators.BinaryClassification.auPR(),
      trainTestEvaluators = Seq(new OpBinaryClassificationEvaluator),
      parallelism = 4,
      modelsAndParameters = models
    ).setInput(survivedNum, fv)
      .getOutput()

    val wf2 = new OpWorkflow().setResultFeatures(pred2)
    wf2.isWorkflowCV shouldBe false
    val model2 = wf2.setReader(simplePassengerForCV).train()
    val data2 = model2.score(keepRawFeatures = false, keepIntermediateFeatures = true)

    val summary1 = model1.modelInsights(pred1)
    log.info("model1.summary: \n{}", summary1)
    val summary2 = model2.modelInsights(pred2)
    log.info("model2.summary: \n{}", summary2)

    summary1.selectedModelInfo.get.validationResults.zip(
      summary2.selectedModelInfo.get.validationResults
    ).forall{ case (v1, v2) =>
        v1.metricValues.asInstanceOf[SingleMetric].value < v2.metricValues.asInstanceOf[SingleMetric].value
    } shouldBe true
  }

  def compare(
    data1: DataFrame,
    data2: DataFrame,
    pred1: FeatureLike[Prediction],
    pred2: FeatureLike[Prediction]
  ): Unit = {
    val winner1 = pred1.originStage.asInstanceOf[ModelSelector[_, _]].bestEstimator.get
    val winner2 = pred2.originStage.asInstanceOf[ModelSelector[_, _]].bestEstimator.get
    winner1.estimator.getClass shouldEqual winner2.estimator.getClass
    val params1 = winner1.estimator.asInstanceOf[PipelineStage].extractParamMap.getAsMap()
    val params2 = winner2.estimator.asInstanceOf[PipelineStage].extractParamMap.getAsMap()
    params1.keySet.union(params2.keySet).foreach{k =>
      if (!Seq("outputFeatureName", "inputFeatures").contains(k)) params1(k) shouldEqual params2(k)
    }
    val d1s = data1.select(pred1.name, KeyFieldName).sort(KeyFieldName).collect()
    val d2s = data2.select(pred2.name, KeyFieldName).sort(KeyFieldName).collect()
    d1s.zip(d2s).foreach {
      case (r1, r2) =>
        math.abs(r1.getMap[String, Double](0)(Prediction.Keys.PredictionName) -
          r2.getMap[String, Double](0)(Prediction.Keys.PredictionName)) should be < 0.5
        if (r1.size > 2) math.abs(r1.getAs[Vector](1)(0) - r2.getAs[Vector](1)(0)) should be < 0.5
    }
  }

}

class Leaker(uid: String = UID[BinaryTransformer[_, _, _]]) extends
  BinaryTransformer[Real, RealNN, RealNN](operationName = "makeLeaker", uid = uid) {
  override def transformFn: (Real, RealNN) => RealNN =
    (f: Real, l: RealNN) => if (l.v.exists(_ > 0)) 1.0.toRealNN else 0.0.toRealNN
  override def outputIsResponse: Boolean = false
}

/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op

import com.salesforce.op.evaluators._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.filters.RawFeatureFilter
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.readers._
import com.salesforce.op.stages.base.unary._
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry._
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.preparators.SanityChecker
import com.salesforce.op.stages.impl.regression.{LossType, RegressionModelSelector, RegressionModelsToTry}
import com.salesforce.op.stages.impl.selector.{ModelSelectorBaseNames, SelectedModel}
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.test.{Passenger, PassengerCSV, PassengerSparkFixtureTest, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.joda.time.{DateTime, Duration}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory

import scala.reflect.runtime.universe.TypeTag

@RunWith(classOf[JUnitRunner])
class OpWorkflowTest extends FlatSpec with PassengerSparkFixtureTest {

  val log = LoggerFactory.getLogger(this.getClass)

  private val density = weight / height
  private val weightNormed = new NormEstimatorTest[Real]().setTest(false).setInput(weight).getOutput()
  private val heightNormed = new NormEstimatorTest[RealNN]().setInput(height).getOutput()
  private val densityByHeightNormed = density * heightNormed
  private val whyNotNormed = new NormEstimatorTest[Real]().setInput(densityByHeightNormed).getOutput()
  private val densityNormed = new NormEstimatorTest[Real]().setInput(density).getOutput()

  private lazy val workflow = new OpWorkflow().setResultFeatures(whyNotNormed, weightNormed)

  private lazy val workflowLocation = tempDir + "/op-workflow-test-model-" + DateTime.now().getMillis
  private lazy val workflowLocation2 = tempDir + "/op-workflow-test-model-2-" + DateTime.now().getMillis
  private lazy val workflowLocation3 = tempDir + "/op-workflow-test-model-3-" + DateTime.now().getMillis
  private lazy val workflowLocation4 = tempDir + "/op-workflow-test-model-4-" + DateTime.now().getMillis

  Spec[OpWorkflow] should "correctly trace the history of stages needed to create the final output" in {
    workflow.getResultFeatures() shouldBe Array(whyNotNormed, weightNormed)

    val stages = workflow.getStages()
    stages.length shouldBe 5
    stages.take(2).toSet shouldBe Set(heightNormed.originStage, density.originStage)
    stages.drop(2).head shouldBe whyNotNormed.parents.head.originStage
    stages.drop(3).toSet shouldBe Set(weightNormed.originStage, whyNotNormed.originStage)
  }

  it should "have all of the inputs set for each stage in the workflow" in {
    val stages = workflow.getStages()
    assert(stages.forall(_.getInputFeatures().nonEmpty))
  }

  it should "throw an error if you try to fit without setting the reader" in {
    intercept[IllegalArgumentException](workflow.train())
  }

  it should "throw an error if you try to set a non serializable stage" in {
    class NotSerializable(val v: Double)
    val ns = new NotSerializable(1.0)
    val weightNotSer =
      weight.transformWith(new UnaryLambdaTransformer[Real, Real]("blarg", v => v.value.map(_ + ns.v).toReal))
    val wf = new OpWorkflow().setReader(dataReader)

    val error = intercept[IllegalArgumentException](wf.setResultFeatures(weightNotSer))
    error.getMessage.contains(weightNotSer.originStage.uid) shouldBe true
  }

  it should "throw an error if you try to re-use a stage" in {
    val stage = new NormEstimatorTest[Real]()
    val densityNormed2 = stage.setInput(density).getOutput()
    val weightNormed2 = stage.setInput(weight).getOutput()
    intercept[IllegalArgumentException](new OpWorkflow().setResultFeatures(whyNotNormed, weightNormed2, densityNormed2))
  }

  it should "throw an error if you try to set a stage with no uid arg in ctor" in {
    val weightNoUid = weight.transformWith(new NoUidTest)
    val wf = new OpWorkflow().setReader(dataReader)

    val error = intercept[IllegalArgumentException](wf.setResultFeatures(weightNoUid))
    error.getMessage.contains(weightNoUid.originStage.uid) shouldBe true
  }

  it should "have a raw filter feature when specified" in {
    val wf = new OpWorkflow()
      .setResultFeatures(whyNotNormed, weightNormed)
      .withRawFeatureFilter(Option(dataReader), None)
    wf.rawFeatureFilter.get.isInstanceOf[RawFeatureFilter[_]] shouldBe true
  }

  it should "correctly remove blacklisted features when possible" in {
    val fv = Seq(age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap).transmogrify()
    val survivedNum = survived.occurs()
    val checked = survivedNum.sanityCheck(fv)
    val (pred, rawPred, prob) = BinaryClassificationModelSelector().setInput(survivedNum, checked).getOutput()
    val wf = new OpWorkflow()
      .setResultFeatures(whyNotNormed, prob)
      .withRawFeatureFilter(Option(dataReader), None)
    wf.rawFeatures should contain theSameElementsAs
      Array(age, boarded, booleanMap, description, gender, height, numericMap, stringMap, survived, weight)

    val blacklist: Array[OPFeature] = Array(age, gender, description, stringMap, numericMap)
    wf.setBlacklist(blacklist)
    wf.getBlacklist() should contain theSameElementsAs blacklist
    wf.rawFeatures should contain theSameElementsAs
      Array(boarded, booleanMap, height, survived, weight)
    wf.getResultFeatures().flatMap(_.rawFeatures).distinct.sortBy(_.name) should contain theSameElementsAs
      Array(boarded, booleanMap, height, survived, weight)
  }

  it should "correctly allow you to interact with updated features when things are blacklisted" in {
    val fv = Seq(age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap).transmogrify()
    val survivedNum = survived.occurs()
    val checked = survivedNum.sanityCheck(fv)
    val (pred, rawPred, prob) = BinaryClassificationModelSelector
      .withTrainValidationSplit(splitter = None, seed = 42, validationMetric = Evaluators.BinaryClassification.error())
      .setInput(survivedNum, checked).getOutput()
    val wf = new OpWorkflow()
      .setResultFeatures(whyNotNormed, prob)
      .withRawFeatureFilter(
        trainingReader = Option(dataReader),
        scoringReader = None,
        minFillRate = 0.7,
        protectedFeatures = Array(height, weight))

    val wfM = wf.train()
    val data = wfM.score()
    data.schema.fields.size shouldBe 3
    val Array(whyNotNormed2, prob2) = wfM.getUpdatedFeatures(Array(whyNotNormed, prob))
    data.select(whyNotNormed2.name, prob2.name).count() shouldBe 6
  }

  it should "throw an error when it is not possible to remove blacklisted features" in {
    val fv = Seq(age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap).transmogrify()
    val survivedNum = survived.occurs()
    val (pred, rawPred, prob) = BinaryClassificationModelSelector().setInput(survivedNum, fv).getOutput()
    val wf = new OpWorkflow()
      .setResultFeatures(whyNotNormed, prob)
      .withRawFeatureFilter(Option(dataReader), None)

    val error = intercept[RuntimeException](
      wf.setBlacklist(Array(age, gender, height, description, stringMap, numericMap))
    )
    error.getMessage.contains("creation of required result feature (height-weight_4-stagesApplied_Real")
  }

  it should "be able to compute a partial dataset in both workflow and workflow model" in {
    val fields =
      List(KeyFieldName, height.name, weight.name, heightNormed.name, density.name,
        densityByHeightNormed.name, whyNotNormed.name)

    val data = workflow.setReader(dataReader).computeDataUpTo(whyNotNormed)
    data.schema.fieldNames should contain theSameElementsAs fields

    val model = workflow.train()
    val dataModel = model.computeDataUpTo(whyNotNormed)
    dataModel.schema.fieldNames should contain theSameElementsAs fields

    model.save(workflowLocation2)
    val loadedModel = workflow.loadModel(workflowLocation2)
    val dataModel2 = loadedModel.setReader(dataReader).computeDataUpTo(whyNotNormed)
    dataModel2.schema.fieldNames should contain theSameElementsAs fields
  }

  it should s"return an ${classOf[OpWorkflowModel].getSimpleName} when it is fit" in {
    val model = workflow.setReader(dataReader).train()

    model.getStages().length shouldBe 5
    model.getResultFeatures() shouldBe workflow.getResultFeatures()
    model.getOriginStageOf(heightNormed).isInstanceOf[UnaryModel[_, _]]

    val metadata = model.getMetadata(weightNormed, heightNormed).mapValues(meta =>
      OpVectorMetadata("outputName", meta)
    )
    val expected = Map(
      weightNormed -> OpVectorMetadata("outputName", Array(NormEstimatorTest.columnMeta),
        Map(weight.name -> weight.history())),
      heightNormed -> OpVectorMetadata("outputName", Array(NormEstimatorTest.columnMeta),
        Map(height.name -> height.history()))
    )

    metadata shouldEqual expected
    model.reader.get shouldBe workflow.reader.get
  }

  it should "use the raw feature filter to generate data instead of the reader when the filter is specified" in {
    val fv = Seq(age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap).transmogrify()
    val survivedNum = survived.occurs()
    val (pred, rawPred, prob) = BinaryClassificationModelSelector().setInput(survivedNum, fv).getOutput()
    val wf = new OpWorkflow()
      .setResultFeatures(pred, rawPred, prob)
      .withRawFeatureFilter( Option(dataReader), Option(simpleReader),
        maxFillRatioDiff = 1.0 ) // only height and the female key of maps should meet this criteria
    val data = wf.computeDataUpTo(weight)

    data.schema.fields.map(_.name).toSet shouldEqual
      Set("key", "height", "survived", "stringMap", "numericMap", "booleanMap")
  }

  it should "return a model that transforms the data correctly" in {
    val model = workflow.setReader(dataReader).train()
    val data = model.score()
    data.schema.fieldNames should contain theSameElementsAs
      (workflow.getResultFeatures().map(_.name) :+ KeyFieldName).distinct

    data.schema.fields.map(_.dataType) should contain only(StringType, DoubleType)

    val partialData = model.computeDataUpTo(density).schema.fieldNames
    List("weight", "height", KeyFieldName).forall(n => partialData.contains(n)) shouldBe true
  }

  it should "leave the intermediate features in the scoring output, if requested to" in {
    val model = workflow.setReader(dataReader).train()
    val data = model.score(keepRawFeatures = false, keepIntermediateFeatures = true)

    data.schema.fieldNames should contain theSameElementsAs
      (workflow.stages.map(_.getOutputFeatureName) :+ KeyFieldName).distinct
  }

  it should "leave the raw features in the scoring output, if requested to" in {
    val model = workflow.setReader(dataReader).train()
    val data = model.score(keepRawFeatures = true, keepIntermediateFeatures = false)

    data.schema.fieldNames should contain theSameElementsAs List(
      KeyFieldName, weight.name, height.name, weightNormed.name,
      whyNotNormed.name)
  }

  it should "leave both the raw & intermediate features in the scoring output, if requested to" in {
    val model = workflow.setReader(dataReader).train()
    val data = model.score(keepRawFeatures = true, keepIntermediateFeatures = true)

    data.schema.fieldNames should contain theSameElementsAs List(
      KeyFieldName, height.name, weight.name, heightNormed.name, density.name, weightNormed.name,
      densityByHeightNormed.name, whyNotNormed.name
    )
  }

  it should "correctly set parameters on workflow stages when the parameter map is set" in {
    workflow.stages.collect {
      case net: NormEstimatorTest[_] => net.getTest
    } should contain theSameElementsAs Array(false, false, false)

    workflow.setParameters(
      OpParams(stageParams = Map("NormEstimatorTest" -> Map("test" -> true), "NotThere" -> Map("test" -> 1)))
    )

    workflow.stages.collect {
      case net: NormEstimatorTest[_] => net.getTest
    } should contain theSameElementsAs Array(true, true, true)

    workflow.setParameters(new OpParams())
  }

  it should "correctly set parameters on the workflow even when the params are not spark params" in {
    val features = Seq(height, weight, gender, age).transmogrify()
    val survivedNum = survived.occurs()
    val (pred, rawPred, prob) =
      BinaryClassificationModelSelector.withCrossValidation(
        seed = 4242,
        splitter = Option(DataBalancer(reserveTestFraction = 0.2, seed = 4242)))
        .setModelsToTry(LogisticRegression)
        .setLogisticRegressionRegParam(0.01)
        .setInput(survivedNum, features)
        .getOutput()
    val newWorkflow = new OpWorkflow().setResultFeatures(features, pred)

    newWorkflow.stages.collect {
      case bc: Stage1BinaryClassificationModelSelector =>
        bc.get(bc.useLR) shouldBe Some(true)
        bc.get(bc.useRF) shouldBe Some(false)
        bc.lRGrid.build().length shouldBe 1
    }

    newWorkflow.setParameters(
      OpParams(stageParams =
        Map("Stage1BinaryClassificationModelSelector" ->
          Map("LogisticRegressionRegParam" -> Seq(0.05, 0.5, 5),
            "ModelsToTry" -> Seq(LogisticRegression, RandomForest)
          ))
      )
    )
    newWorkflow.stages.collect {
      case bc: Stage1BinaryClassificationModelSelector =>
        bc.get(bc.useLR) shouldBe Some(true)
        bc.get(bc.useRF) shouldBe Some(true)
        bc.lRGrid.build().length shouldBe 3
    }

  }

  it should "allow addition of features to a fitted workflow by producing a new workflow with the fitted stages" in {
    val model = workflow.setReader(dataReader).train()
    val densityNormed = new NormEstimatorTest[Real]().setInput(density).getOutput()
    val newWorkflow = new OpWorkflow().setResultFeatures(densityNormed).setReader(dataReader)
    newWorkflow.withModelStages(model)
    log.info(model.getStages().map(s => s.uid + s.getClass.getSimpleName).toList.mkString)
    log.info(newWorkflow.getStages().map(s => s.uid + s.getClass.getSimpleName).toList.mkString)
    (densityNormed.originStage +: model.getStages()).toSet.diff(newWorkflow.getStages().toSet) shouldBe Set.empty
  }

  it should "allow addition of features and produce the same results" in {
    val model = workflow.setReader(dataReader).train()
    val scoresDF = model.score(keepRawFeatures = true, keepIntermediateFeatures = true)
    val oldScores = scoresDF.collect(height, weight, heightNormed, density)
    val newWorkflow = new OpWorkflow().setResultFeatures(densityNormed).setReader(dataReader)
      .withModelStages(model).train()
    val newScoresDF = newWorkflow.score(keepRawFeatures = true, keepIntermediateFeatures = true)
    val newScored = newScoresDF.collect(height, weight, heightNormed, density, densityNormed)
    oldScores.map(_._1).toSet shouldBe newScored.map(_._1).toSet
    oldScores.map(_._2).toSet shouldBe newScored.map(_._2).toSet
    oldScores.map(_._3).toSet shouldBe newScored.map(_._3).toSet
    oldScores.head.productIterator.length + 1 shouldBe newScored.head.productIterator.length
  }

  it should "fallback to reading from a reader path when the path in the params is not set" in {
    val testReader = DataReaders.Simple.avro[Passenger](path = Option("dummy"))
    workflow.setReader(testReader).setParameters(new OpParams())
    // we expect this to fail, since it's a 'dummy' path
    intercept[IllegalArgumentException](workflow.train())
  }

  it should "fail to read when no path is set" in {
    val testReader = DataReaders.Simple.avro[Passenger](path = None)
    workflow.setReader(testReader).setParameters(new OpParams())
    val thrown = intercept[IllegalArgumentException](workflow.train())
    thrown.getMessage shouldBe "requirement failed: The path is not set"
  }

  it should "print correct summary information when used with estimators containing summaries" in {
    val features = Seq(height, weight, gender, age).transmogrify()
    val survivedNum = survived.occurs()
    val checked = new SanityChecker()
      .setCheckSample(1.0)
      .setInput(survivedNum, features)
      .getOutput()
    val (pred, rawPred, prob) =
      BinaryClassificationModelSelector.withCrossValidation(
        seed = 4242,
        splitter = Option(DataBalancer(reserveTestFraction = 0.2, seed = 4242)))
        .setModelsToTry(LogisticRegression)
        .setLogisticRegressionRegParam(0.01, 0.1)
        .setInput(survivedNum, checked)
        .getOutput()
    val newWorkflow = new OpWorkflow().setResultFeatures(features, pred).setReader(dataReader)
    val fittedWorkflow = newWorkflow.train()

    if (log.isInfoEnabled) fittedWorkflow.score(keepRawFeatures = true).show()

    val summary = fittedWorkflow.summary()
    log.info(summary)
    summary.contains(classOf[SanityChecker].getSimpleName) shouldBe true
    summary.contains("logreg") shouldBe true
    summary.contains(""""regParam" : "0.1"""") shouldBe true
    summary.contains(""""regParam" : "0.01"""") shouldBe true
    summary.contains(ModelSelectorBaseNames.HoldOutEval) shouldBe true
    summary.contains(ModelSelectorBaseNames.TrainingEval) shouldBe true
  }

  it should "be able to refit a workflow with calibrated probability" in {
    val features = Seq(height, weight, gender, age, stringMap, genderPL).transmogrify()
    val survivedNum = survived.occurs()
    val checked = new SanityChecker().setCheckSample(1.0).setInput(survivedNum, features).getOutput()
    val (pred, rawPred, prob) =
      BinaryClassificationModelSelector
        .withCrossValidation(seed = 42, splitter = None)
        .setModelsToTry(LogisticRegression)
        .setLogisticRegressionRegParam(0.01, 0.1)
        .setInput(survivedNum, checked)
        .getOutput()
    val newWorkflow = new OpWorkflow().setResultFeatures(pred).setReader(dataReader)
    val fittedWorkflow = newWorkflow.train()
    fittedWorkflow.save(workflowLocation3)
    val loadedWorkflow = newWorkflow.loadModel(workflowLocation3)
    val probability = prob.map[RealNN](_.v(0).toRealNN)
    val calibrated = probability.toPercentile()
    val calibratedWorkflow = new OpWorkflow().setResultFeatures(calibrated).setReader(dataReader)
    val newTrained = calibratedWorkflow.withModelStages(loadedWorkflow).train()
    val scores = newTrained.score()

    val calib = scores.collect(calibrated)

    calib.length shouldBe 6
    calib.forall(_.v.exists(n => n >= 0.0 && n <= 99.0))
  }

  it should "have the same metadata and scores with all scoring methods and the same metrics when expected" in {
    val features = Seq(height, weight, gender, age).transmogrify()
    val survivedNum = survived.occurs()
    val checked = new SanityChecker().setCheckSample(1.0).setInput(survivedNum, features).getOutput()
    val (pred, rawPred, prob) =
      BinaryClassificationModelSelector
        .withCrossValidation(seed = 42, splitter = None)
        .setModelsToTry(LogisticRegression)
        .setLogisticRegressionRegParam(0.01, 0.1)
        .setInput(survivedNum, checked)
        .getOutput()
    val probability = prob.map[RealNN](_.v(0).toRealNN)
    val calibrated = probability.toPercentile()
    val newWorkflow = new OpWorkflow().setResultFeatures(pred, calibrated).setReader(dataReader)
    val fittedWorkflow = newWorkflow.train()
    val evaluator = Evaluators.BinaryClassification()
      .setRawPredictionCol(rawPred)
      .setLabelCol(survivedNum)
      .setPredictionCol(pred)
      .setProbabilityCol(prob)

    val scores1 = fittedWorkflow.score(keepIntermediateFeatures = true)
    val (scores2, metrics) = fittedWorkflow.scoreAndEvaluate(evaluator = evaluator, keepIntermediateFeatures = true)

    val fields1 = scores1.schema.fields
    val fields2 = scores2.schema.fields

    scores1.collect().sortBy(_.getAs[String](DataFrameFieldNames.KeyFieldName)) should contain theSameElementsAs
      scores2.collect().sortBy(_.getAs[String](DataFrameFieldNames.KeyFieldName))
    scores1.schema.fields.map(_.name) should contain theSameElementsAs scores2.schema.fields.map(_.name)
    scores1.schema.fields.map(_.metadata.toString()) should contain theSameElementsAs
      scores2.schema.fields.map(_.metadata.toString())

    val probs = scores2.collect(prob)
    val thresholds = probs.map(_.value(1)).distinct.sorted.reverse

    metrics shouldBe BinaryClassificationMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 5.0, 0.0, 0.0,
      thresholds.toSeq, Seq(1.0, 0.5, 0.25, 0.2, 1.0/6), Seq(1.0, 1.0, 1.0, 1.0, 1.0),
      Seq(0.0, 0.2, 0.6, 0.8, 1.0))
  }

  it should "return an empty data set if passed empty data for scoring" in {
    val model = new OpWorkflow().setResultFeatures(whyNotNormed, weightNormed).setReader(dataReader).train()
    val emptyReader = new CustomReader[Passenger](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Passenger], Dataset[Passenger]] =
        Left(spark.sparkContext.emptyRDD[Passenger])
    }
    val scores = model.setReader(emptyReader).score()
    scores.collect().length shouldBe 0
  }

  it should "eliminate duplicate stages for result features of the same level" in {
    val weightNormedPlusTwo = weightNormed + 2
    val weightNormedPlusThree = weightNormed + 3
    val workflow = new OpWorkflow().setResultFeatures(weightNormedPlusTwo, weightNormedPlusThree)
    workflow.getStages() shouldBe Array(
      weightNormed.originStage,
      weightNormedPlusTwo.originStage,
      weightNormedPlusThree.originStage
    )
  }

  it should "work with data passed in RDD rather than DataReader" in {
    val (ds, f1, f2, f3) = TestFeatureBuilder(Seq(
      (1.0.toReal, 2.0.toReal, 3.0.toReal),
      (1.0.toReal, 2.0.toReal, 3.0.toReal),
      (1.0.toReal, 2.0.toReal, 3.0.toReal)
    ))
    val rdd = ds.rdd
    val f = (f1 + f2 + f3).fillMissingWithMean().zNormalize()
    val wf = new OpWorkflow().setResultFeatures(f).setInputRDD(rdd)
    val modelLocation = checkpointDir + "/setInputRDD"
    wf.train().save(modelLocation)
    val scores = wf.loadModel(modelLocation).setInputRDD(rdd).score()
    scores.collect(f) shouldEqual Seq.fill(3)(0.0.toRealNN)
  }

  it should "work with data passed in Dataset rather than DataReader" in {
    val (ds, f1, f2, f3) = TestFeatureBuilder(Seq(
      (1.0.toReal, 2.0.toReal, 3.0.toReal),
      (1.0.toReal, 2.0.toReal, 3.0.toReal),
      (1.0.toReal, 2.0.toReal, 3.0.toReal)
    ))
    val f = (f1 + f2 + f3).fillMissingWithMean().zNormalize()
    val wf = new OpWorkflow().setResultFeatures(f).setInputDataset(ds)
    val modelLocation = checkpointDir + "/setInputDataset"
    wf.train().save(modelLocation)
    val scores = wf.loadModel(modelLocation).setInputDataset(ds).score()
    scores.collect(f) shouldEqual Seq.fill(3)(0.0.toRealNN)
  }

}

class NoUidTest extends UnaryTransformer[Real, Real]("blarg", UID[NoUidTest]) {
  def transformFn: Real => Real = identity
}

class NormEstimatorTest[I <: Real](uid: String = UID[NormEstimatorTest[_]])
  (implicit tti: TypeTag[I], ttiv: TypeTag[I#Value])
  extends UnaryEstimator[I, Real](operationName = "minMaxNorm", uid = uid) {

  def fitFn(dataset: Dataset[I#Value]): UnaryModel[I, Real] = {
    val grouped = dataset.groupBy()
    val maxVal = grouped.max().first().getDouble(0)
    val minVal = grouped.min().first().getDouble(0)
    new NormEstimatorTestModel[I](min = minVal, max = maxVal, operationName = operationName, uid = uid)
  }

  val test = new BooleanParam(this, "test", "test")
  setDefault(test, false)
  def setTest(value: Boolean): this.type = set(test, value)
  def getTest: Boolean = $(test)

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val hist = Map(in1.name -> FeatureHistory(originFeatures = in1.originFeatures, stages = in1.stages))
    setMetadata(OpVectorMetadata("outputName", Array(NormEstimatorTest.columnMeta), hist).toMetadata)
  }
}

final class NormEstimatorTestModel[I <: Real] private[op]
(
  val min: Double,
  val max: Double,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[I])
  extends UnaryModel[I, Real](operationName = operationName, uid = uid) {
  def transformFn: I => Real = _.v.map(v => (v - min) / (max - min)).toReal
}

object NormEstimatorTest {

  val columnMeta = OpVectorColumnMetadata(
    parentFeatureName = Seq("parentFeature"),
    parentFeatureType = Seq(FeatureTypeDefaults.Real.getClass.getName),
    indicatorGroup = Some("indicator_group"),
    indicatorValue = None
  )

}

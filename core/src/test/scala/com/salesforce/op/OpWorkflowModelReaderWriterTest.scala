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

import java.io.File

import com.salesforce.op.OpWorkflowModelReadWriteShared.FieldNames._
import com.salesforce.op.features.OPFeature
import com.salesforce.op.features.types.{Real, RealNN}
import com.salesforce.op.filters._
import com.salesforce.op.readers.{AggregateAvroReader, DataReaders}
import com.salesforce.op.stages.OPStage
import com.salesforce.op.stages.sparkwrappers.generic.SwUnaryEstimator
import com.salesforce.op.test.{Passenger, PassengerSparkFixtureTest}
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.joda.time.DateTime
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, Formats, JArray, JValue}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterEach, FlatSpec}
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class OpWorkflowModelReaderWriterTest
  extends FlatSpec with UIDReset with PassengerSparkFixtureTest with BeforeAndAfterEach {

  implicit val jsonFormats: Formats = DefaultFormats
  val log = LoggerFactory.getLogger(this.getClass)

  val workflowParams = OpParams(
    stageParams = Map("a" -> Map("aa" -> 1, "aaa" -> 2), "b" -> Map("bb" -> 3, "bbb" -> 4)),
    readerParams = Map("test" -> new ReaderParams(Some("a"), Some(3), Map.empty))
  )
  var saveFlowPath: String = _
  var saveModelPath: String = _

  val saveFlowPathStable: String = tempDir + "/op-rw-wf-test-" + DateTime.now().getMillis

  override protected def beforeEach(): Unit = {
    super.beforeEach()
    saveFlowPath = tempDir + "/op-rw-wf-test-" + DateTime.now().getMillis
    saveModelPath = tempDir + "/op-rw-wf-model-test-" + DateTime.now().getMillis
  }

  override def afterAll: Unit = {
    super.afterAll
    deleteRecursively(new File(saveFlowPathStable))
  }

  // dummy data source
  val dummyReader: AggregateAvroReader[Passenger] = DataReaders.Aggregate.avro[Passenger](
    path = Some(""),
    key = _.getPassengerId.toString,
    aggregateParams = null
  )

  val distributions = Array(FeatureDistribution("a", None, 1L, 1L, Array(1.0), Array(1.0)),
    FeatureDistribution("b", Option("b"), 2L, 2L, Array(2.0), Array(2.0)))

  val rawFeatureFilterResults = RawFeatureFilterResults(
    rawFeatureDistributions = distributions
  )


  def makeDummyModel(wf: OpWorkflow): OpWorkflowModel = {
    val model = new OpWorkflowModel(wf.uid, wf.parameters)
      .setStages(wf.stages)
      .setFeatures(wf.resultFeatures)
      .setParameters(wf.parameters)
      .setRawFeatureFilterResults(rawFeatureFilterResults)

    model.setReader(wf.reader.get)
  }

  def makeModelAndJson(wf: OpWorkflow): (OpWorkflowModel, JValue) = {
    val wfM = makeDummyModel(wf)
    val json = OpWorkflowModelWriter.toJson(wfM, saveModelPath)
    log.info(json)
    (wfM, parse(json))
  }

  trait SingleStageFlow {
    val density = weight / height
    val wf = new OpWorkflow()
      .setReader(dummyReader)
      .setResultFeatures(density)
      .setParameters(workflowParams)
      .setRawFeatureFilterResults(rawFeatureFilterResults)
    val (wfM, jsonModel) = makeModelAndJson(wf)
  }

  trait MultiStageFlow {
    val density = weight / height
    val weight2 = density * height
    val dummy = height * height // dead branch
    val wf = new OpWorkflow()
      .setReader(dummyReader)
      .setResultFeatures(density, weight2)
      .setParameters(workflowParams)
      .setRawFeatureFilterResults(rawFeatureFilterResults)
    val (wfM, jsonModel) = makeModelAndJson(wf)
  }

  trait RawFeatureFlow {
    val wf = new OpWorkflow()
      .setReader(dummyReader)
      .setResultFeatures(weight)
      .setParameters(workflowParams)
      .setRawFeatureFilterResults(rawFeatureFilterResults)
    val (wfM, jsonModel) = makeModelAndJson(wf)
  }

  trait SwSingleStageFlow {
    val scaler = new StandardScaler().setWithStd(false).setWithMean(false)
    val swEstimator = new SwUnaryEstimator[RealNN, RealNN, StandardScalerModel, StandardScaler](
      inputParamName = "foo",
      outputParamName = "foo2",
      operationName = "foo3",
      sparkMlStageIn = Some(scaler)
    )
    val scaled = height.transformWith(swEstimator)
    val wf = new OpWorkflow()
      .setParameters(workflowParams)
      .setReader(dummyReader)
      .setResultFeatures(scaled)
      .setRawFeatureFilterResults(rawFeatureFilterResults)
    val (wfM, jsonModel) = makeModelAndJson(wf)
  }

  "Single Stage OpWorkflowWriter" should "have proper json entries" in new SingleStageFlow {
    val modelKeys = jsonModel.extract[Map[String, Any]].keys
    modelKeys should contain theSameElementsAs OpWorkflowModelReadWriteShared.FieldNames.values.map(_.entryName)
  }

  it should "have correct result id" in new SingleStageFlow {
    val idsM = (jsonModel \ ResultFeaturesUids.entryName).extract[Array[String]]
    idsM should contain theSameElementsAs Array(density.uid)
  }

  it should "have a single stage" in new SingleStageFlow {
    val stagesM = (jsonModel \ Stages.entryName).extract[JArray]
    stagesM.values.size shouldBe 1
  }

  it should "have 3 features" in new SingleStageFlow {
    val featsM = (jsonModel \ AllFeatures.entryName).extract[JArray]
    featsM.values.size shouldBe 3
  }

  it should "have correct uid" in new SingleStageFlow {
    val uidM = (jsonModel \ Uid.entryName).extract[String]
    uidM shouldBe wf.uid
  }

  it should "have correct parameters" in new SingleStageFlow {
    val paramsM = OpParams.fromString((jsonModel \ Parameters.entryName).extract[String]).get
    paramsM.readerParams.toString() shouldBe workflowParams.readerParams.toString()
    paramsM.stageParams shouldBe workflowParams.stageParams
  }

  "MultiStage OpWorkflowWriter" should "recover all relevant stages" in new MultiStageFlow {
    val stagesM = (jsonModel \ Stages.entryName).extract[JArray]
    stagesM.values.size shouldBe 2
  }

  it should "recover all relevant features" in new MultiStageFlow {
    val featsM = (jsonModel \ AllFeatures.entryName).extract[JArray]
    featsM.values.size shouldBe 4
  }

  it should "have the correct results feature ids" in new MultiStageFlow {
    val idsM = (jsonModel \ ResultFeaturesUids.entryName).extract[Array[String]]
    idsM should contain theSameElementsAs Array(density.uid, weight2.uid)
  }

  "Raw feature only OpWorkflowWriter" should "recover no stages" in new RawFeatureFlow {
    val stagesM = (jsonModel \ Stages.entryName).extract[JArray]
    stagesM.values.length shouldBe 0
  }

  it should "recover raw feature in feature list" in new RawFeatureFlow {
    val featsM = (jsonModel \ AllFeatures.entryName).extract[JArray]
    featsM.values.size shouldBe 1
  }

  it should "have the correct results feature ids" in new RawFeatureFlow {
    val idsM = (jsonModel \ ResultFeaturesUids.entryName).extract[Array[String]]
    idsM should contain theSameElementsAs Array(weight.uid)
  }

  Spec[OpWorkflowModelReader] should "load proper single stage workflow" in new SingleStageFlow {
    wfM.save(saveModelPath)
    val wfMR = wf.loadModel(saveModelPath)
    compareWorkflowModels(wfMR, wfM)
  }

  it should "load proper multiple stage workflow" in new MultiStageFlow {
    wfM.save(saveModelPath)
    val wfMR = wf.loadModel(saveModelPath)
    compareWorkflowModels(wfMR, wfM)
  }

  it should "load proper raw feature workflow" in new RawFeatureFlow {
    wfM.save(saveModelPath)
    val wfMR = wf.loadModel(saveModelPath)
    compareWorkflowModels(wfMR, wfM)
  }

  it should "load proper workflow with spark wrapped stages" in new SwSingleStageFlow {
    wfM.save(saveModelPath)
    val wfMR = wf.loadModel(saveModelPath)
    compareWorkflowModels(wfMR, wfM)
  }

  it should "work for models" in new SingleStageFlow {
    wf.setReader(dataReader)
    val model = wf.train()
    model.save(saveFlowPath)
    val wfMR = wf.loadModel(saveFlowPath)
    compareWorkflowModels(model, wfMR)
  }

  trait VectorizedFlow extends UIDReset {
    val cat = Seq(gender, boarded, height, age, description).transmogrify()
    val catHead = cat.map[Real](v => Real(v.value.toArray.headOption))
    val wf = new OpWorkflow()
      .setParameters(workflowParams)
      .setResultFeatures(catHead)
  }

  it should "load workflow model with vectorized feature" in new VectorizedFlow {
    wf.setReader(dataReader)
    val wfM = wf.train()
    wfM.save(saveFlowPath)
    val wfMR = wf.loadModel(saveFlowPath)
    compareWorkflowModels(wfMR, wfM)
  }

  it should "save a workflow model that has a RawFeatureFilter" in new VectorizedFlow {
    wf.withRawFeatureFilter(Some(dataReader), None, minFillRate = 0.8)
    val wfM = wf.train()
    wfM.save(saveFlowPathStable)
    wf.getBlacklist().map(_.name) should contain theSameElementsAs Array("age", "description")
    val wfMR = wf.loadModel(saveFlowPathStable)
    wfMR.getBlacklist().map(_.name) should contain theSameElementsAs Array("age", "description")
    compareWorkflowModels(wfM, wfMR)
  }

  it should "load a workflow model that has a RawFeatureFilter and a different workflow" in new VectorizedFlow {
    val wfM = wf.loadModel(saveFlowPathStable)
    wf.getResultFeatures().head.name shouldBe wfM.getResultFeatures().head.name
    wf.getResultFeatures().head.history().originFeatures should contain theSameElementsAs
      Array("age", "boarded", "description", "gender", "height")
    wfM.getResultFeatures().head.history().originFeatures should contain theSameElementsAs
      Array("boarded", "gender", "height")
    wfM.getBlacklist().map(_.name) should contain theSameElementsAs Array("age", "description")
  }

  it should "load model and allow copying it" in new VectorizedFlow {
    val wfM = wf.loadModel(saveFlowPathStable)
    val copy = wfM.copy()
    copy.uid shouldBe wfM.uid
    copy.trainingParams.toString shouldBe wfM.trainingParams.toString
    copy.isWorkflowCV shouldBe wfM.isWorkflowCV
    copy.reader shouldBe wfM.reader
    copy.resultFeatures shouldBe wfM.resultFeatures
    copy.rawFeatures shouldBe wfM.rawFeatures
    copy.blacklistedFeatures shouldBe wfM.blacklistedFeatures
    copy.blacklistedMapKeys shouldBe wfM.blacklistedMapKeys
    copy.rawFeatureFilterResults shouldBe wfM.rawFeatureFilterResults
    copy.stages.map(_.uid) shouldBe wfM.stages.map(_.uid)
    copy.parameters.toString shouldBe wfM.parameters.toString
  }

  it should "be able to load a old version of a saved model" in new VectorizedFlow {
    val wfM = wf.loadModel("src/test/resources/OldModelVersion")
    wfM.getBlacklist().isEmpty shouldBe true
  }

  it should "be able to load a old version of a saved model (v0.5.1)" in new VectorizedFlow {
    // note: in these old models, raw feature filter config will be set to the config defaults
    // but we never re-initialize raw feature filter when loading a model (only scoring, no training)
    val wfM = wf.loadModel("src/test/resources/OldModelVersion_0_5_1")
    wfM.getRawFeatureFilterResults().rawFeatureFilterMetrics shouldBe empty
    wfM.getRawFeatureFilterResults().exclusionReasons shouldBe empty
  }

  it should "error on loading a model without workflow" in {
    val error = intercept[RuntimeException](OpWorkflowModel.load(saveFlowPathStable))
    error.getMessage should startWith("Failed to load Workflow from path")
    error.getCause.isInstanceOf[NotImplementedError] shouldBe true
    error.getCause.getMessage shouldBe "Loading models without the original workflow is currently not supported"
  }

  def compareFeatures(f1: Array[OPFeature], f2: Array[OPFeature]): Unit = {
    f1.length shouldBe f2.length
    f1.sortBy(_.uid) should contain theSameElementsAs f2.sortBy(_.uid)
  }

  // Ordering of stages is important
  def compareStages(stages1: Array[OPStage], stages2: Array[OPStage]): Unit = {
    stages1.length shouldBe stages2.length
    stages1.zip(stages2).foreach {
      case (s1, s2) => {
        s1.uid shouldBe s2.uid
        compareFeatures(s1.getInputFeatures(), s2.getInputFeatures())

        val s1Feats: Array[OPFeature] = Array(s1.getOutput())
        val s2Feats: Array[OPFeature] = Array(s2.getOutput())
        compareFeatures(s1Feats, s2Feats)
      }
    }
  }

  def compareWorkflows(wf1: OpWorkflow, wf2: OpWorkflow): Unit = {
    wf1.uid shouldBe wf2.uid
    compareParams(wf1.parameters, wf2.parameters)
    compareFeatures(wf1.resultFeatures, wf2.resultFeatures)
    compareFeatures(wf1.blacklistedFeatures, wf2.blacklistedFeatures)
    compareFeatures(wf1.rawFeatures, wf2.rawFeatures)
    compareStages(wf1.stages, wf2.stages)
    RawFeatureFilterResultsComparison.compare(wf1.getRawFeatureFilterResults(), wf2.getRawFeatureFilterResults())
  }

  def compareWorkflowModels(wf1: OpWorkflowModel, wf2: OpWorkflowModel): Unit = {
    wf1.uid shouldBe wf2.uid
    compareParams(wf1.trainingParams, wf2.trainingParams)
    compareParams(wf1.parameters, wf2.parameters)
    compareFeatures(wf1.resultFeatures, wf2.resultFeatures)
    compareFeatures(wf1.blacklistedFeatures, wf2.blacklistedFeatures)
    compareFeatures(wf1.rawFeatures, wf2.rawFeatures)
    compareStages(wf1.stages, wf2.stages)
    RawFeatureFilterResultsComparison.compare(wf1.getRawFeatureFilterResults(), wf2.getRawFeatureFilterResults())
  }

  def compareParams(p1: OpParams, p2: OpParams): Unit = {
    p1.stageParams shouldBe p2.stageParams
    p1.readerParams.toString() shouldBe p2.readerParams.toString()
    p1.customParams shouldBe p2.customParams
  }
}

trait UIDReset {
  UID.reset()
}

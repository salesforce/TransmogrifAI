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

import com.salesforce.op.features.types._
import com.salesforce.op.features.{Feature, FeatureDistributionType}
import com.salesforce.op.filters._
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.preparators._
import com.salesforce.op.stages.impl.regression.{OpLinearRegression, OpXGBoostRegressor, RegressionModelSelector}
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.EstimatorType
import com.salesforce.op.stages.impl.selector.SelectedModel
import com.salesforce.op.stages.impl.selector.ValidationType._
import com.salesforce.op.stages.impl.tuning.{DataCutter, DataSplitter}
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.junit.runner.RunWith
import org.scalactic.Equality
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class ModelInsightsTest extends FlatSpec with PassengerSparkFixtureTest {

  implicit val doubleEquality = new Equality[Double] {
    def areEqual(a: Double, b: Any): Boolean = b match {
      case s: Double => (a.isNaN && s.isNaN) || (a == b)
      case _ => false
    }
  }

  implicit val doubleOptEquality = new Equality[Option[Double]] {
    def areEqual(a: Option[Double], b: Any): Boolean = b match {
      case None => a.isEmpty
      case Some(d: Double) => (a.exists(_.isNaN) && d.isNaN) || a.contains(d)
      case _ => false
    }
  }

  private val density = weight / height
  private val generVec = genderPL.vectorize(topK = 10, minSupport = 1, cleanText = true)
  private val descrVec = description.vectorize(10, false, 1, true)
  private val features = Seq(density, age, generVec, weight, descrVec).transmogrify()
  private val featuresWithMaps = Seq(density, age, generVec, weight, descrVec, numericMap).transmogrify()
  private val label = survived.occurs()
  private val checked =
    label.sanityCheck(features, removeBadFeatures = true, removeFeatureGroup = false, checkSample = 1.0)
  private val checkedWithMaps =
    label.sanityCheck(featuresWithMaps, removeBadFeatures = true, removeFeatureGroup = false, checkSample = 1.0)

  val lr = new OpLogisticRegression()
  val lrParams = new ParamGridBuilder().addGrid(lr.regParam, Array(0.01, 0.1)).build()
  val models = Seq(lr -> lrParams).asInstanceOf[Seq[(EstimatorType, Array[ParamMap])]]

  val xgbClassifier = new OpXGBoostClassifier().setSilent(1).setSeed(42L)
  val xgbRegressor = new OpXGBoostRegressor().setSilent(1).setSeed(42L)
  val xgbClassifierPred = xgbClassifier.setInput(label, features).getOutput()
  val xgbRegressorPred = xgbRegressor.setInput(label, features).getOutput()
  lazy val xgbWorkflow =
    new OpWorkflow().setResultFeatures(xgbClassifierPred, xgbRegressorPred).setReader(dataReader)
  lazy val xgbWorkflowModel = xgbWorkflow.train()

  val pred = BinaryClassificationModelSelector
    .withCrossValidation(seed = 42, splitter = Option(DataSplitter(seed = 42, reserveTestFraction = 0.1)),
      modelsAndParameters = models)
    .setInput(label, checked)
    .getOutput()

  val predWithMaps = BinaryClassificationModelSelector
    .withCrossValidation(seed = 42, splitter = Option(DataSplitter(seed = 42, reserveTestFraction = 0.1)),
      modelsAndParameters = models)
    .setInput(label, checkedWithMaps)
    .getOutput()

  val predLin = RegressionModelSelector
    .withTrainValidationSplit(seed = 42, dataSplitter = None, modelsAndParameters = Seq(new OpLinearRegression() ->
      new ParamGridBuilder().build()).asInstanceOf[Seq[(EstimatorType, Array[ParamMap])]])
    .setInput(label, features)
    .getOutput()

  val params = new OpParams()

  lazy val workflow = new OpWorkflow().setResultFeatures(predLin, pred).setParameters(params).setReader(dataReader)

  lazy val workflowModel = workflow.train()

  lazy val modelWithRFF = new OpWorkflow()
    .setResultFeatures(predWithMaps)
    .setParameters(params)
    .withRawFeatureFilter(Option(dataReader), Option(simpleReader), bins = 10, minFillRate = 0.0,
      maxFillDifference = 1.0, maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 1.0, maxCorrelation = 0.4)
    .train()

  val rawNames = Set(age.name, weight.name, height.name, genderPL.name, description.name)

  Spec[ModelInsights] should "throw an error when you try to get insights on a raw feature" in {
    val ex = the[IllegalArgumentException] thrownBy {
      workflowModel.modelInsights(age)
    }
    val expectedErrorMessage = "eature.? '?age.* is either a raw feature or not part of this workflow ?model"
    ex.getMessage.toLowerCase should include regex expectedErrorMessage
  }

  it should "return empty insights when no selector, label, feature vector, or model are found" in {
    val insights = workflowModel.modelInsights(density)
    insights.label.labelName shouldBe None
    insights.features.isEmpty shouldBe true
    insights.selectedModelInfo.isEmpty shouldBe true
    insights.trainingParams shouldEqual params
    insights.stageInfo.keys.head shouldEqual s"${density.originStage.operationName}_${density.originStage.uid}"
  }

  it should "return only feature insights when no selector, label, or model are found" in {
    val insights = workflowModel.modelInsights(features)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    insights.label.labelName shouldBe None
    insights.features.size shouldBe 5
    insights.features.map(_.featureName).toSet shouldEqual rawNames
    ageInsights.derivedFeatures.size shouldBe 2
    genderInsights.derivedFeatures.size shouldBe 4
    insights.selectedModelInfo.isEmpty shouldBe true
    insights.trainingParams shouldEqual params
    insights.stageInfo.keys.size shouldEqual 8
  }

  it should "return model insights even when correlation is turned off for some features" in {
    val featuresFinal = Seq(
      description.vectorize(numHashes = 10, autoDetectLanguage = false, minTokenLength = 1, toLowercase = true),
      stringMap.vectorize(cleanText = true, numHashes = 10)
    ).combine()
    val featuresChecked = label.sanityCheck(featuresFinal, correlationExclusion = CorrelationExclusion.HashedText)
    val prediction = MultiClassificationModelSelector
      .withCrossValidation(seed = 42, splitter = Option(DataCutter(seed = 42, reserveTestFraction = 0.1)),
        modelsAndParameters = models)
      .setInput(label, featuresChecked)
      .getOutput()
    val workflow = new OpWorkflow().setResultFeatures(prediction).setParameters(params).setReader(dataReader)
    val workflowModel = workflow.train()
    val insights = workflowModel.modelInsights(prediction)
    insights.features.size shouldBe 2
    insights.features.flatMap(_.derivedFeatures).size shouldBe 23
  }

  it should "return feature insights with selector info and label info even when no models are found" in {
    val insights = workflowModel.modelInsights(checked)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    insights.label.labelName shouldBe Some(label.name)
    insights.label.distribution.get.isInstanceOf[Continuous] shouldBe true
    insights.label.rawFeatureName shouldBe Seq(survived.name)
    insights.label.rawFeatureType shouldBe Seq(survived.typeName)
    insights.label.stagesApplied.size shouldBe 1
    insights.label.sampleSize shouldBe Some(5.0)
    insights.features.size shouldBe 5
    insights.features.map(_.featureName).toSet shouldEqual rawNames
    ageInsights.derivedFeatures.size shouldBe 2
    ageInsights.derivedFeatures.foreach { f =>
      f.contribution shouldBe Seq.empty
      f.corr.nonEmpty shouldBe true
      f.variance.nonEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    genderInsights.derivedFeatures.size shouldBe 4
    genderInsights.derivedFeatures.foreach { f =>
      f.contribution shouldBe Seq.empty
      f.corr.nonEmpty shouldBe true
      f.variance.nonEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    insights.selectedModelInfo.isEmpty shouldBe true
    insights.trainingParams shouldEqual params
    insights.stageInfo.keys.size shouldEqual 10
  }

  it should "find the sanity checker metadata even if the model has been serialized" in {
    val path = tempDir.toString + "/model-insights-test-" + System.currentTimeMillis()
    val json = OpWorkflowModelWriter.toJson(workflowModel, path)
    val loadedModel = new OpWorkflowModelReader(workflow).loadJson(json, path)
    val insights = loadedModel.get.modelInsights(checked)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    ageInsights.derivedFeatures.foreach { f =>
      f.contribution shouldBe Seq.empty
      f.corr.nonEmpty shouldBe true
      f.variance.nonEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    genderInsights.derivedFeatures.foreach { f =>
      f.contribution shouldBe Seq.empty
      f.corr.nonEmpty shouldBe true
      f.variance.nonEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
  }

  it should "return feature insights with selector info and label info and model info" in {
    val insights = workflowModel.modelInsights(pred)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    insights.label.labelName shouldBe Some(label.name)
    insights.label.distribution.get.isInstanceOf[Continuous] shouldBe true
    insights.label.rawFeatureName shouldBe Seq(survived.name)
    insights.label.rawFeatureType shouldBe Seq(survived.typeName)
    insights.label.stagesApplied.size shouldBe 1
    insights.label.sampleSize shouldBe Some(5.0)
    insights.features.size shouldBe 5
    insights.features.map(_.featureName).toSet shouldEqual rawNames
    ageInsights.derivedFeatures.size shouldBe 2
    ageInsights.derivedFeatures(0).contribution.size shouldBe 1
    ageInsights.derivedFeatures(1).contribution.size shouldBe 0
    ageInsights.derivedFeatures.foreach { f =>
      f.corr.nonEmpty shouldBe true
      f.variance.nonEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    genderInsights.derivedFeatures.size shouldBe 4
    genderInsights.derivedFeatures.foreach { f =>
      if (f.excluded.contains(true)) f.contribution.size shouldBe 0 else f.contribution.size shouldBe 1
      f.corr.nonEmpty shouldBe true
      f.variance.nonEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    insights.selectedModelInfo.get.validationType shouldBe CrossValidation
    insights.trainingParams shouldEqual params
    insights.stageInfo.keys.size shouldEqual 11
  }

  it should "return feature insights with label info and model info even when no sanity checker is found" in {
    val insights = workflowModel.modelInsights(predLin)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    insights.label.labelName shouldBe Some(label.name)
    insights.label.distribution.isEmpty shouldBe true
    insights.label.rawFeatureName shouldBe Seq(survived.name)
    insights.label.rawFeatureType shouldBe Seq(survived.typeName)
    insights.label.stagesApplied.size shouldBe 1
    insights.label.sampleSize.isEmpty shouldBe true
    insights.features.size shouldBe 5
    insights.features.map(_.featureName).toSet shouldEqual rawNames
    ageInsights.derivedFeatures.size shouldBe 2
    ageInsights.derivedFeatures.foreach { f =>
      f.contribution.size shouldBe 1
      f.corr.isEmpty shouldBe true
      f.variance.isEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    genderInsights.derivedFeatures.size shouldBe 4
    genderInsights.derivedFeatures.foreach { f =>
      f.contribution.size shouldBe 1
      f.corr.isEmpty shouldBe true
      f.variance.isEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    insights.selectedModelInfo.get.validationType shouldBe TrainValidationSplit
    insights.trainingParams shouldEqual params
    insights.stageInfo.keys.size shouldEqual 10
  }

  it should "correctly pull out model contributions when passed a selected model" in {
    val reg = ModelInsights.getModelContributions(
      Option(workflowModel.getOriginStageOf(predLin).asInstanceOf[SelectedModel])
    )
    val lin = ModelInsights.getModelContributions(
      Option(workflowModel.getOriginStageOf(pred).asInstanceOf[SelectedModel])
    )
    reg.size shouldBe 1
    reg.head.size shouldBe 21

    lin.size shouldBe 1
    lin.head.size shouldBe OpVectorMetadata("", checked.originStage.getMetadata()).columns.length
  }

  it should "pretty print" in {
    val insights = workflowModel.modelInsights(pred)
    insights.selectedModelInfo.isDefined shouldBe true
    val pretty = insights.prettyPrint()
    val modelType = BinaryClassificationModelsToTry.OpLogisticRegression
    val sm = insights.selectedModelInfo.get
    sm.bestModelType shouldBe modelType.toString
    sm.validationResults.size shouldBe 2

    pretty should include(s"Selected Model - $modelType")
    withClue("include only best model info: ") {
      pretty should include(sm.bestModelUID)
      pretty should include(sm.bestModelType)
      pretty should include(sm.bestModelName)
    }
    withClue("not include other models info: ") {
      val others = sm.validationResults.filterNot(v =>
        v.modelUID == sm.bestModelUID && v.modelName == sm.bestModelName && v.modelType == sm.bestModelType
      )
      others.size shouldBe 1
      others.foreach { m =>
        pretty should not include m.modelName
      }
    }
    pretty should include regex raw"area under precision-recall\s+|\s+1.0"
    pretty should include("Model Evaluation Metrics")
    pretty should include("Top Model Insights")
    pretty should include("Top Positive Correlations")
    pretty should include("Top Contributions")
  }

  it should "correctly serialize and deserialize from json when raw feature filter is not used" in {
    val insights = workflowModel.modelInsights(pred)
    ModelInsights.fromJson(insights.toJson()) match {
      case Failure(e) => fail(e)
      case Success(deser) =>
        insights.label shouldEqual deser.label
        insights.features.zip(deser.features).foreach{
          case (i, o) =>
            i.featureName shouldEqual o.featureName
            i.featureType shouldEqual o.featureType
            i.derivedFeatures.zip(o.derivedFeatures).foreach{ case (ii, io) => ii.corr shouldEqual io.corr }
        }
        RawFeatureFilterResultsComparison.compareSeqDistributions(
          insights.rawFeatureFilterResults.rawFeatureDistributions,
          deser.rawFeatureFilterResults.rawFeatureDistributions
        )
        RawFeatureFilterResultsComparison.compareSeqExclusionReasons(
          insights.rawFeatureFilterResults.exclusionReasons,
          deser.rawFeatureFilterResults.exclusionReasons
        )
        insights.selectedModelInfo.toSeq.zip(deser.selectedModelInfo.toSeq).foreach{
          case (o, i) =>
            o.validationType shouldEqual i.validationType
            o.validationParameters.keySet shouldEqual i.validationParameters.keySet
            o.dataPrepParameters.keySet shouldEqual i.dataPrepParameters.keySet
            o.dataPrepResults shouldEqual i.dataPrepResults
            o.evaluationMetric shouldEqual i.evaluationMetric
            o.problemType shouldEqual i.problemType
            o.bestModelUID shouldEqual i.bestModelUID
            o.bestModelName shouldEqual i.bestModelName
            o.bestModelType shouldEqual i.bestModelType
            o.validationResults.zip(i.validationResults).foreach{
              case (ov, iv) => ov.metricValues shouldEqual iv.metricValues
                ov.modelParameters.keySet shouldEqual iv.modelParameters.keySet
            }
            o.trainEvaluation shouldEqual i.trainEvaluation
            o.holdoutEvaluation shouldEqual o.holdoutEvaluation
        }
        insights.trainingParams.toJson() shouldEqual deser.trainingParams.toJson()
        insights.stageInfo.keys shouldEqual deser.stageInfo.keys

    }
  }

  it should "correctly serialize and deserialize from json when raw feature filter is used" in {
    val insights = modelWithRFF.modelInsights(predWithMaps)
    ModelInsights.fromJson(insights.toJson()) match {
      case Failure(e) => fail(e)
      case Success(deser) =>
        insights.label shouldEqual deser.label
        insights.features.zip(deser.features).foreach {
          case (i, o) =>
            i.featureName shouldEqual o.featureName
            i.featureType shouldEqual o.featureType
            i.derivedFeatures.zip(o.derivedFeatures).foreach { case (ii, io) => ii.corr shouldEqual io.corr }
        }
        RawFeatureFilterResultsComparison.compareSeqDistributions(
          insights.rawFeatureFilterResults.rawFeatureDistributions,
          deser.rawFeatureFilterResults.rawFeatureDistributions
        )
        RawFeatureFilterResultsComparison.compareSeqExclusionReasons(
          insights.rawFeatureFilterResults.exclusionReasons,
          deser.rawFeatureFilterResults.exclusionReasons
        )
        insights.selectedModelInfo.toSeq.zip(deser.selectedModelInfo.toSeq).foreach {
          case (o, i) =>
            o.validationType shouldEqual i.validationType
            o.validationParameters.keySet shouldEqual i.validationParameters.keySet
            o.dataPrepParameters.keySet shouldEqual i.dataPrepParameters.keySet
            o.dataPrepResults shouldEqual i.dataPrepResults
            o.evaluationMetric shouldEqual i.evaluationMetric
            o.problemType shouldEqual i.problemType
            o.bestModelUID shouldEqual i.bestModelUID
            o.bestModelName shouldEqual i.bestModelName
            o.bestModelType shouldEqual i.bestModelType
            o.validationResults.zip(i.validationResults).foreach {
              case (ov, iv) => ov.metricValues shouldEqual iv.metricValues
                ov.modelParameters.keySet shouldEqual iv.modelParameters.keySet
            }
            o.trainEvaluation shouldEqual i.trainEvaluation
            o.holdoutEvaluation shouldEqual o.holdoutEvaluation
        }
        insights.trainingParams.toJson() shouldEqual deser.trainingParams.toJson()
        insights.stageInfo.keys shouldEqual deser.stageInfo.keys
    }
  }

  it should "have feature insights for features that are removed by the raw feature filter" in {
    val insights = modelWithRFF.modelInsights(predWithMaps)

    modelWithRFF.blacklistedFeatures should contain theSameElementsAs Array(age, description, genderPL, weight)
    val heightIn = insights.features.find(_.featureName == age.name).get
    heightIn.derivedFeatures.size shouldBe 1
    heightIn.derivedFeatures.head.excluded shouldBe Some(true)

    modelWithRFF.blacklistedMapKeys should contain theSameElementsAs Map(numericMap.name -> Set("Female"))
    val mapDerivedIn = insights.features.find(_.featureName == numericMap.name).get.derivedFeatures
    val droppedMapDerivedIn = mapDerivedIn.filter(_.derivedFeatureName == "Female")
    mapDerivedIn.size shouldBe 3
    droppedMapDerivedIn.size shouldBe 1
    droppedMapDerivedIn.head.excluded shouldBe Some(true)
    droppedMapDerivedIn.head.derivedFeatureGroup shouldBe Some("Female")
  }

  val labelName = "l"

  val summary = SanityCheckerSummary(
    correlationsWLabel = Correlations(Seq("f0_f0_f2_1", "f0_f0_f3_2"), Seq(5.2, 5.3), Seq("f1_0"),
      CorrelationType.Pearson),
    dropped = Seq("f1_0"),
    featuresStatistics = SummaryStatistics(count = 3, sampleFraction = 0.01, max = Seq(0.1, 0.2, 0.3, 0.0),
      min = Seq(1.1, 1.2, 1.3, 1.0), mean = Seq(2.1, 2.2, 2.3, 2.0), variance = Seq(3.1, 3.2, 3.3, 3.0)),
    names = Seq("f1_0", "f0_f0_f2_1", "f0_f0_f3_2", labelName),
    categoricalStats = Array(
      CategoricalGroupStats(
        group = "f0_f0_f2",
        categoricalFeatures = Array("f0_f0_f2_1"),
        contingencyMatrix = Map("0" -> Array(13.0, 17.0), "1" -> Array(5.0, 15.0), "2" -> Array(14.0, 36.0)),
        cramersV = 6.2,
        pointwiseMutualInfo = Map("0" -> Array(7.2), "1" -> Array(8.2), "2" -> Array(9.2)),
        mutualInfo = 10.2,
        maxRuleConfidences = Array(0.0),
        supports = Array(1.0)
      ), CategoricalGroupStats(
        group = "f0_f0_f2",
        categoricalFeatures = Array( "f0_f0_f3_2"),
        contingencyMatrix = Map("0" -> Array(11.0, 12.0), "1" -> Array(12.0, 12.0), "2" -> Array(13.0, 12.0)),
        cramersV = 6.3,
        pointwiseMutualInfo = Map("0" -> Array(7.3), "1" -> Array(8.3), "2" -> Array(9.3)),
        mutualInfo = 10.3,
        maxRuleConfidences = Array(0.0),
        supports = Array(1.0)
      )
    )
  )

  val lbl = Feature[RealNN](labelName, true, null, Seq(), "test")
  val f1 = Feature[Real]("f1", true, null, Seq(), "test")
  val f0 = Feature[PickList]("f0", true, null, Seq(), "test")

  val meta = OpVectorMetadata(
    "fv",
    OpVectorColumnMetadata(
      parentFeatureName = Seq("f1"),
      parentFeatureType = Seq(classOf[Real].getName),
      grouping = None,
      indicatorValue = None
    ) +: Array("f2", "f3").map { name =>
      OpVectorColumnMetadata(
        parentFeatureName = Seq("f0"),
        parentFeatureType = Seq(classOf[PickList].getName),
        grouping = Option("f0"),
        indicatorValue = Option(name)
      )
    },
    Seq("f1", "f0").map(name => name -> FeatureHistory(originFeatures = Seq(name), stages = Seq())).toMap
  )

  it should "correctly extract the LabelSummary from the label and sanity checker info" in {
    val labelSum = ModelInsights.getLabelSummary(Option(lbl), Option(summary))
    labelSum.labelName shouldBe Some(labelName)
    labelSum.rawFeatureName shouldBe lbl.history().originFeatures
    labelSum.rawFeatureType shouldBe Seq(classOf[RealNN].getName)
    labelSum.stagesApplied shouldBe lbl.history().stages
    labelSum.sampleSize shouldBe Some(3.0)
    labelSum.distribution.get.isInstanceOf[Discrete] shouldBe true
    labelSum.distribution.get.asInstanceOf[Discrete].domain should contain theSameElementsAs Array("0", "1", "2")
    labelSum.distribution.get.asInstanceOf[Discrete].prob should contain theSameElementsAs Array(0.3, 0.2, 0.5)
  }

  it should "correctly extract the FeatureInsights from the sanity checker summary and vector metadata" in {
    val featureInsights = ModelInsights.getFeatureInsights(
      Option(meta), Option(summary), None, Array(f1, f0), Array.empty, Map.empty[String, Set[String]]
    )
    featureInsights.size shouldBe 2

    val f1In = featureInsights.find(_.featureName == "f1").get
    f1In.featureType shouldBe classOf[Real].getName
    f1In.derivedFeatures.size shouldBe 1

    val f1InDer = f1In.derivedFeatures.head
    f1InDer.derivedFeatureName shouldBe "f1_0"
    f1InDer.stagesApplied shouldBe Seq.empty
    f1InDer.derivedFeatureGroup shouldBe None
    f1InDer.derivedFeatureValue shouldBe None
    f1InDer.excluded shouldBe Option(true)
    f1InDer.corr.map(_.toString) shouldBe Some("NaN")
    f1InDer.cramersV shouldBe None
    f1InDer.mutualInformation shouldBe None
    f1InDer.pointwiseMutualInformation shouldBe Map.empty
    f1InDer.countMatrix shouldBe Map.empty
    f1InDer.contribution shouldBe Seq.empty
    f1InDer.min shouldBe Some(1.1)
    f1InDer.max shouldBe Some(0.1)
    f1InDer.mean shouldBe Some(2.1)
    f1InDer.variance shouldBe Some(3.1)

    val f0In = featureInsights.find(_.featureName == "f0").get
    f0In.featureName shouldBe "f0"
    f0In.featureType shouldBe classOf[PickList].getName
    f0In.derivedFeatures.size shouldBe 2

    val f0InDer2 = f0In.derivedFeatures.head
    f0InDer2.derivedFeatureName shouldBe "f0_f0_f2_1"
    f0InDer2.stagesApplied shouldBe Seq.empty
    f0InDer2.derivedFeatureGroup shouldBe Some("f0")
    f0InDer2.derivedFeatureValue shouldBe Some("f2")
    f0InDer2.excluded shouldBe Option(false)
    f0InDer2.corr shouldBe Some(5.2)
    f0InDer2.cramersV shouldBe Some(6.2)
    f0InDer2.mutualInformation shouldBe Some(10.2)
    f0InDer2.pointwiseMutualInformation shouldBe Map("0" -> 7.2, "1" -> 8.2, "2" -> 9.2)
    f0InDer2.countMatrix shouldBe Map("0" -> 13.0, "1" -> 5.0, "2" -> 14.0)
    f0InDer2.contribution shouldBe Seq.empty
    f0InDer2.min shouldBe Some(1.2)
    f0InDer2.max shouldBe Some(0.2)
    f0InDer2.mean shouldBe Some(2.2)
    f0InDer2.variance shouldBe Some(3.2)

    val f0InDer3 = f0In.derivedFeatures.last
    f0InDer3.derivedFeatureName shouldBe "f0_f0_f3_2"
    f0InDer3.stagesApplied shouldBe Seq.empty
    f0InDer3.derivedFeatureGroup shouldBe Some("f0")
    f0InDer3.derivedFeatureValue shouldBe Some("f3")
    f0InDer3.excluded shouldBe Option(false)
    f0InDer3.corr shouldBe Some(5.3)
    f0InDer3.cramersV shouldBe Some(6.3)
    f0InDer3.mutualInformation shouldBe Some(10.3)
    f0InDer3.pointwiseMutualInformation shouldBe Map("0" -> 7.3, "1" -> 8.3, "2" -> 9.3)
    f0InDer3.countMatrix shouldBe Map("0" -> 11.0, "1" -> 12.0, "2" -> 13.0)
    f0InDer3.contribution shouldBe Seq.empty
    f0InDer3.min shouldBe Some(1.3)
    f0InDer3.max shouldBe Some(0.3)
    f0InDer3.mean shouldBe Some(2.3)
    f0InDer3.variance shouldBe Some(3.3)
  }

  it should "include raw feature distribution information when RawFeatureFilter is used" in {
    val wfRawFeatureDistributions = modelWithRFF.getRawFeatureDistributions()

    val wfDistributionsGrouped = wfRawFeatureDistributions.groupBy(_.name)

    val trainingDistributions = modelWithRFF.getRawTrainingFeatureDistributions()
    trainingDistributions.foreach(_.`type` shouldBe FeatureDistributionType.Training)

    val scoringDistributions = modelWithRFF.getRawScoringFeatureDistributions()
    scoringDistributions.foreach(_.`type` shouldBe FeatureDistributionType.Scoring)

    trainingDistributions ++ scoringDistributions shouldBe wfRawFeatureDistributions

    /**
     * Currently, raw features that aren't explicitly blacklisted, but are not used because they are inputs to
     * explicitly blacklisted features are not present as raw features in the model, nor in ModelInsights. For example,
     * weight is explicitly blacklisted here, which means that height will not be added as a raw feature even though
     * it's not explicitly blacklisted itself.
     */
    val insights = modelWithRFF.modelInsights(predWithMaps)

    val insightsDistributionsGrouped = insights.rawFeatureFilterResults.rawFeatureDistributions.groupBy(_.name)

    insightsDistributionsGrouped.keys.foreach( name =>
      insightsDistributionsGrouped(name) shouldBe wfDistributionsGrouped.getOrElse(name, Seq.empty)
    )
  }

  it should "not include raw feature distribution information when RawFeatureFilter is not used" in {
    val insights = workflowModel.modelInsights(pred)
    insights.rawFeatureFilterResults.rawFeatureDistributions shouldBe Seq.empty
  }

  it should "return model insights for xgboost classification" in {
    noException should be thrownBy xgbWorkflowModel.modelInsights(xgbClassifierPred)
    val insights = xgbWorkflowModel.modelInsights(xgbClassifierPred)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    insights.features.size shouldBe 5
    insights.features.map(_.featureName).toSet shouldEqual rawNames
    ageInsights.derivedFeatures.size shouldBe 2
    ageInsights.derivedFeatures.foreach { f =>
      f.contribution.size shouldBe 1
      f.corr.isEmpty shouldBe true
      f.variance.isEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    genderInsights.derivedFeatures.size shouldBe 4
    genderInsights.derivedFeatures.foreach { f =>
      f.contribution.size shouldBe 1
      f.corr.isEmpty shouldBe true
      f.variance.isEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
  }

  it should "return model insights for xgboost regression" in {
    noException should be thrownBy xgbWorkflowModel.modelInsights(xgbRegressorPred)
    val insights = xgbWorkflowModel.modelInsights(xgbRegressorPred)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    insights.features.size shouldBe 5
    insights.features.map(_.featureName).toSet shouldEqual rawNames
    ageInsights.derivedFeatures.size shouldBe 2
    ageInsights.derivedFeatures.foreach { f =>
      f.contribution.size shouldBe 1
      f.corr.isEmpty shouldBe true
      f.variance.isEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
    genderInsights.derivedFeatures.size shouldBe 4
    genderInsights.derivedFeatures.foreach { f =>
      f.contribution.size shouldBe 1
      f.corr.isEmpty shouldBe true
      f.variance.isEmpty shouldBe true
      f.cramersV.isEmpty shouldBe true
    }
  }
}

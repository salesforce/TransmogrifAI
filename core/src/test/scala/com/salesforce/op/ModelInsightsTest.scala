/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.features.Feature
import com.salesforce.op.features.types.{FeatureTypeDefaults, PickList, Real, RealNN}
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry.LogisticRegression
import com.salesforce.op.stages.impl.preparators._
import com.salesforce.op.stages.impl.regression.RegressionModelSelector
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry.LinearRegression
import com.salesforce.op.stages.impl.selector.SelectedModel
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class ModelInsightsTest extends FlatSpec with PassengerSparkFixtureTest {

  private val density = weight / height
  private val generVec = genderPL.vectorize(topK = 10, minSupport = 1, cleanText = true)
  private val descrVec = description.vectorize(10, false, 1, true)
  private val features = Seq(density, age, generVec, weight, descrVec).transmogrify()
  private val label = survived.occurs()
  private val checked = label.sanityCheck(features, removeBadFeatures = true, checkSample = 1.0)

  val (pred, rawPred, prob) = BinaryClassificationModelSelector
    .withCrossValidation(seed = 42, splitter = None)
    .setModelsToTry(LogisticRegression)
    .setLogisticRegressionRegParam(0.01, 0.1)
    .setInput(label, checked)
    .getOutput()

  val predLin = RegressionModelSelector
    .withTrainValidationSplit(seed = 42, dataSplitter = None)
    .setModelsToTry(LinearRegression)
    .setInput(label, features)
    .getOutput()

  val params = new OpParams()

  lazy val workflow = new OpWorkflow().setResultFeatures(predLin, prob).setParameters(params).setReader(dataReader)

  lazy val workflowModel = workflow.train()

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
    insights.stageInfo.keys.size shouldEqual 6
  }

  it should "return feature insights with selector info and label info even when models are found" in {
    val insights = workflowModel.modelInsights(checked)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    insights.label.labelName shouldBe Some(label.name)
    insights.label.distribution.get.isInstanceOf[Continuous] shouldBe true
    insights.label.rawFeatureName shouldBe Seq(survived.name)
    insights.label.rawFeatureType shouldBe Seq(survived.typeName)
    insights.label.stagesApplied.size shouldBe 1
    insights.label.sampleSize shouldBe Some(6.0)
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
    insights.stageInfo.keys.size shouldEqual 8
  }

  it should "return feature insights with selector info and label info and model info" in {
    val insights = workflowModel.modelInsights(prob)
    val ageInsights = insights.features.filter(_.featureName == age.name).head
    val genderInsights = insights.features.filter(_.featureName == genderPL.name).head
    insights.label.labelName shouldBe Some(label.name)
    insights.label.distribution.get.isInstanceOf[Continuous] shouldBe true
    insights.label.rawFeatureName shouldBe Seq(survived.name)
    insights.label.rawFeatureType shouldBe Seq(survived.typeName)
    insights.label.stagesApplied.size shouldBe 1
    insights.label.sampleSize shouldBe Some(6.0)
    insights.features.size shouldBe 5
    insights.features.map(_.featureName).toSet shouldEqual rawNames
    ageInsights.derivedFeatures.size shouldBe 2
    ageInsights.derivedFeatures.foreach { f =>
      f.contribution.size shouldBe 1
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
    insights.selectedModelInfo.contains("crossValidationResults") shouldBe true
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
    insights.selectedModelInfo.contains("trainValidationSplitResults") shouldBe true
    insights.trainingParams shouldEqual params
    insights.stageInfo.keys.size shouldEqual 8
  }

  it should "correctly pull out model contributions when passed a selected model" in {
    val reg = ModelInsights.getModelContributions(
      Option(workflowModel.getOriginStageOf(predLin).asInstanceOf[SelectedModel])
    )
    val lin = ModelInsights.getModelContributions(
      Option(workflowModel.getOriginStageOf(pred).asInstanceOf[SelectedModel])
    )
    reg.size shouldBe 1
    reg.head.size shouldBe 20

    lin.size shouldBe 1
    lin.head.size shouldBe OpVectorMetadata("", checked.originStage.getMetadata()).columns.length
  }

  it should "correctly serialize and deserialize from json" in {
    val insights = workflowModel.modelInsights(prob)
    ModelInsights.fromJson(insights.toJson()) match {
      case Failure(e) => fail(e)
      case Success(deser) =>
        insights.label.labelName shouldEqual deser.label.labelName
        insights.features.length shouldEqual deser.features.length
        insights.selectedModelInfo.keys shouldEqual deser.selectedModelInfo.keys
        insights.trainingParams.toJson() shouldEqual deser.trainingParams.toJson()
        insights.stageInfo.keys shouldEqual deser.stageInfo.keys
    }
  }

  val labelName = "l"

  val summary = SanityCheckerSummary(
    correlationsWLabel = Correlations(Seq("f0_f0_f2_1", "f0_f0_f3_2"), Seq(5.2, 5.3), Seq("f1_0"),
      CorrelationType.Pearson),
    dropped = Seq("f1_0"),
    featuresStatistics = SummaryStatistics(count = 3, sampleFraction = 0.01, max = Seq(0.1, 0.2, 0.3, 0.0),
      min = Seq(1.1, 1.2, 1.3, 1.0), mean = Seq(2.1, 2.2, 2.3, 2.0), variance = Seq(3.1, 3.2, 3.3, 3.0),
      numNull = Seq(4.1, 4.2, 4.3, 4.0)),
    names = Seq("f1_0", "f0_f0_f2_1", "f0_f0_f3_2", labelName),
    categoricalStats = CategoricalStats(
      categoricalFeatures = Array("f0_f0_f2_1", "f0_f0_f3_2"),
      cramersVs = Array(6.2, 6.3),
      pointwiseMutualInfos = Map("0" -> Array(7.2, 7.3), "1" -> Array(8.2, 8.3), "2" -> Array(9.2, 9.3)),
      mutualInfos = Array(10.2, 10.3),
      counts = Map("0" -> Array(11.2, 11.3, 11.0), "1" -> Array(12.2, 12.3, 12.0), "2" -> Array(13.2, 13.3, 13.0))
    )
  )

  val lbl = Feature[RealNN](labelName, true, null, Seq(), "test")
  val f1 = Feature[Real]("f1", true, null, Seq(), "test")
  val f0 = Feature[PickList]("f0", true, null, Seq(), "test")

  val meta = OpVectorMetadata(
    "fv",
    OpVectorColumnMetadata(
      parentFeatureName = Seq("f1"),
      parentFeatureType = Seq(FeatureTypeDefaults.Real.getClass.getName),
      indicatorGroup = None,
      indicatorValue = None
    ) +: Array("f2", "f3").map { name =>
      OpVectorColumnMetadata(
        parentFeatureName = Seq("f0"),
        parentFeatureType = Seq(FeatureTypeDefaults.PickList.getClass.getName),
        indicatorGroup = Option("f0"),
        indicatorValue = Option(name)
      )
    },
    Seq("f1", "f0").map(name => name -> FeatureHistory(originFeatures = Seq(name), stages = Seq())).toMap
  )

  it should "correctly extract the LabelSummary from the label and sanity checker info" in {
    val labelSum = ModelInsights.getLabelSummary(Option(lbl), Option(summary))
    labelSum.labelName shouldBe Some(labelName)
    labelSum.rawFeatureName shouldBe lbl.history().originFeatures
    labelSum.rawFeatureType shouldBe Seq(FeatureTypeDefaults.RealNN.getClass.getName)
    labelSum.stagesApplied shouldBe lbl.history().stages
    labelSum.sampleSize shouldBe Some(3.0)
    labelSum.distribution.get.isInstanceOf[Discrete] shouldBe true
    labelSum.distribution.get.asInstanceOf[Discrete].domain should contain theSameElementsAs Array("0", "1", "2")
    labelSum.distribution.get.asInstanceOf[Discrete].prob should contain theSameElementsAs Array(11.0, 12.0, 13.0)
  }

  it should "correctly extract the FeatureInsights from the sanity checker summary and vector metadata" in {
    val featureInsights = ModelInsights.getFeatureInsights(Option(meta), Option(summary), None, Array(f1, f0))
    featureInsights.size shouldBe 2

    val f1In = featureInsights.find(_.featureName == "f1").get
    f1In.featureType shouldBe FeatureTypeDefaults.Real.getClass.getName
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
    f1InDer.numberOfNulls shouldBe Some(4.1)

    val f0In = featureInsights.find(_.featureName == "f0").get
    f0In.featureName shouldBe "f0"
    f0In.featureType shouldBe FeatureTypeDefaults.PickList.getClass.getName
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
    f0InDer2.countMatrix shouldBe Map("0" -> 11.2, "1" -> 12.2, "2" -> 13.2)
    f0InDer2.contribution shouldBe Seq.empty
    f0InDer2.min shouldBe Some(1.2)
    f0InDer2.max shouldBe Some(0.2)
    f0InDer2.mean shouldBe Some(2.2)
    f0InDer2.variance shouldBe Some(3.2)
    f0InDer2.numberOfNulls shouldBe Some(4.2)

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
    f0InDer3.countMatrix shouldBe Map("0" -> 11.3, "1" -> 12.3, "2" -> 13.3)
    f0InDer3.contribution shouldBe Seq.empty
    f0InDer3.min shouldBe Some(1.3)
    f0InDer3.max shouldBe Some(0.3)
    f0InDer3.mean shouldBe Some(2.3)
    f0InDer3.variance shouldBe Some(3.3)
    f0InDer3.numberOfNulls shouldBe Some(4.3)
  }

}

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

import com.salesforce.op.evaluators._
import com.salesforce.op.features.types.{Real, _}
import com.salesforce.op.features.{Feature, FeatureDistributionType, FeatureLike}
import com.salesforce.op.filters._
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.feature.{CombinationStrategy, TextStats}
import com.salesforce.op.stages.impl.preparators._
import com.salesforce.op.stages.impl.regression.{OpLinearRegression, OpXGBoostRegressor, RegressionModelSelector}
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.EstimatorType
import com.salesforce.op.stages.impl.selector.ValidationType._
import com.salesforce.op.stages.impl.selector.{SelectedCombinerModel, SelectedModel, SelectedModelCombiner}
import com.salesforce.op.stages.impl.tuning.{DataCutter, DataSplitter}
import com.salesforce.op.test.{PassengerSparkFixtureTest, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomReal
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import ml.dmlc.xgboost4j.scala.spark.OpXGBoostQuietLogging
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.junit.runner.RunWith
import com.twitter.algebird.Moments
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class ModelInsightsTest extends FlatSpec with PassengerSparkFixtureTest with DoubleEquality with OpXGBoostQuietLogging {

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

  val xgbClassifier = new OpXGBoostClassifier().setMissing(0.0f).setSilent(1).setSeed(42L)
  val xgbRegressor = new OpXGBoostRegressor().setMissing(0.0f).setSilent(1).setSeed(42L)
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


  val smallFeatureVariance = 10.0
  val mediumFeatureVariance = 1.0
  val bigFeatureVariance = 100.0
  val smallNorm = RandomReal.normal[Real](0.0, smallFeatureVariance).limit(1000)
  val mediumNorm = RandomReal.normal[Real](10, mediumFeatureVariance).limit(1000)
  val bigNorm = RandomReal.normal[Real](10000.0, bigFeatureVariance).limit(1000)
  val noise = RandomReal.normal[Real](0.0, 100.0).limit(1000)
  // make a simple linear combination of the features (with noise), pass through sigmoid function and binarize
  // to make labels for logistic reg toy data
  def binarize(x: Double): Int = {
    val sigmoid = 1.0 / (1.0 + math.exp(-x))
    if (sigmoid > 0.5) 1 else 0
  }
  val logisticRegLabel = (smallNorm, mediumNorm, noise)
    .zipped.map(_.toDouble.get * 10 + _.toDouble.get + _.toDouble.get).map(binarize(_)).map(RealNN(_))
  // toy label for linear reg is a sum of two scaled Normals, hence we also know its standard deviation
  val linearRegLabel = (smallNorm, bigNorm)
    .zipped.map(_.toDouble.get * 5000 + _.toDouble.get).map(RealNN(_))
  val labelStd = math.sqrt(5000 * 5000 * smallFeatureVariance + bigFeatureVariance)
  def twoFeatureDF(feature1: List[Real], feature2: List[Real], label: List[RealNN]):
  (Feature[RealNN], FeatureLike[OPVector], DataFrame) = {
    val generatedData = feature1.zip(feature2).zip(label).map {
      case ((f1, f2), label) => (f1, f2, label)
    }
    val (rawDF, raw1, raw2, rawLabel) = TestFeatureBuilder("feature1", "feature2", "label", generatedData)
    val labelData = rawLabel.copy(isResponse = true)
    val featureVector = raw1
      .vectorize(fillValue = 0, fillWithMean = true, trackNulls = false, others = Array(raw2))
    val checkedFeatures = labelData.sanityCheck(featureVector, removeBadFeatures = false)
    return (labelData, checkedFeatures, rawDF)
  }

  val linRegDF = twoFeatureDF(smallNorm, bigNorm, linearRegLabel)
  val logRegDF = twoFeatureDF(smallNorm, mediumNorm, logisticRegLabel)

  val unstandardizedLinpred = new OpLinearRegression().setStandardization(false)
    .setInput(linRegDF._1, linRegDF._2).getOutput()

  val standardizedLinpred = new OpLinearRegression().setStandardization(true)
    .setInput(linRegDF._1, linRegDF._2).getOutput()

  val unstandardizedLogpred = new OpLogisticRegression().setStandardization(false)
    .setInput(logRegDF._1, logRegDF._2).getOutput()

  val standardizedLogpred = new OpLogisticRegression().setStandardization(true)
    .setInput(logRegDF._1, logRegDF._2).getOutput()

  def getFeatureImp(standardizedModel: FeatureLike[Prediction],
    unstandardizedModel: FeatureLike[Prediction], DF: DataFrame): Array[Double] = {
    lazy val workFlow = new OpWorkflow()
      .setResultFeatures(standardizedModel, unstandardizedModel).setInputDataset(DF)
    lazy val model = workFlow.train()
    val unstandardizedFtImp = model.modelInsights(unstandardizedModel)
      .features.map(_.derivedFeatures.map(_.contribution))
    val standardizedFtImp = model.modelInsights(standardizedModel)
      .features.map(_.derivedFeatures.map(_.contribution))
    val descaledsmallCoeff = standardizedFtImp.flatten.flatten.head
    val originalsmallCoeff = unstandardizedFtImp.flatten.flatten.head
    val descaledbigCoeff = standardizedFtImp.flatten.flatten.last
    val orginalbigCoeff = unstandardizedFtImp.flatten.flatten.last
    return Array(descaledsmallCoeff, originalsmallCoeff, descaledbigCoeff, orginalbigCoeff)
  }

  def getFeatureMomentsAndCard(inputModel: FeatureLike[Prediction],
    DF: DataFrame): (Map[String, Moments], Map[String, TextStats]) = {
    lazy val workFlow = new OpWorkflow().setResultFeatures(inputModel).setInputDataset(DF)
    lazy val dummyReader = workFlow.getReader()
    lazy val workFlowRFF = workFlow.withRawFeatureFilter(Some(dummyReader), None)
    lazy val model = workFlowRFF.train()
    val insights = model.modelInsights(inputModel)
    val featureMoments = insights.features.map(f => f.featureName -> f.distributions.head.moments.get).toMap
    val featureCardinality = insights.features.map(f => f.featureName -> f.distributions.head.cardEstimate.get).toMap
    featureMoments -> featureCardinality
  }

  val params = new OpParams()

  lazy val workflow = new OpWorkflow().setResultFeatures(predLin, pred)
    .setParameters(params).setReader(dataReader)

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

    // head will be RFF so accessing 2nd element
    insights.stageInfo.keys.slice(1, 2).toList.head shouldEqual
      s"${density.originStage.operationName}_${density.originStage.uid}"

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
    insights.stageInfo.keys.size shouldEqual 9
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
    insights.stageInfo.keys.size shouldEqual 11
  }

  it should "find the sanity checker metadata even if the model has been serialized" in {
    val path = tempDir.toString + "/model-insights-test-" + System.currentTimeMillis()
    val json = OpWorkflowModelWriter.toJson(workflowModel, path)
    val loadedModel = new OpWorkflowModelReader(Some(workflow)).loadJson(json, path)
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
    insights.stageInfo.keys.size shouldEqual 12
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
    insights.stageInfo.keys.size shouldEqual 11
  }

  it should "correctly pull out model contributions when passed a selected model" in {
    val predLinMod = workflowModel.getOriginStageOf(predLin).asInstanceOf[SelectedModel]
    val reg = ModelInsights.getModelContributions(Option(predLinMod))

    val linMod = workflowModel.getOriginStageOf(pred).asInstanceOf[SelectedModel]
    val lin = ModelInsights.getModelContributions(Option(linMod))
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
        insights.features.zip(deser.features).foreach {
          case (i, o) =>
            i.featureName shouldEqual o.featureName
            i.featureType shouldEqual o.featureType
            i.derivedFeatures.zip(o.derivedFeatures).foreach { case (ii, io) => ii.corr shouldEqual io.corr }
            RawFeatureFilterResultsComparison.compareSeqMetrics(i.metrics, o.metrics)
            RawFeatureFilterResultsComparison.compareSeqDistributions(i.distributions, o.distributions)
            RawFeatureFilterResultsComparison.compareSeqExclusionReasons(i.exclusionReasons, o.exclusionReasons)
        }
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
            i.distributions.foreach { i => i.cardEstimate should not be None}
            o.distributions.foreach { o => o.cardEstimate shouldEqual None}
            RawFeatureFilterResultsComparison.compareSeqMetrics(i.metrics, o.metrics)
            RawFeatureFilterResultsComparison.compareSeqDistributions(i.distributions, o.distributions)
            RawFeatureFilterResultsComparison.compareSeqExclusionReasons(i.exclusionReasons, o.exclusionReasons)
        }
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

        // check that raw feature filter config is correctly serialized and deserialized

        def getRawFeatureFilterConfig(modelInsights: ModelInsights): Map[String, String] = {
          modelInsights.stageInfo(RawFeatureFilter.stageName) match {
            case configInfo: Map[String, Map[String, String]]@unchecked =>
              configInfo.getOrElse("params", Map.empty[String, String])
            case _ => Map.empty[String, String]
          }
        }

        (getRawFeatureFilterConfig(insights), getRawFeatureFilterConfig(deser)) match {
          case (paramsMapI, paramsMapD) =>
            paramsMapI.keys shouldEqual paramsMapD.keys
            paramsMapI("minFill") shouldEqual paramsMapD("minFill")
            paramsMapI("maxFillDifference") shouldEqual paramsMapD("maxFillDifference")
            paramsMapI("maxFillRatioDiff") shouldEqual paramsMapD("maxFillRatioDiff")
            paramsMapI("maxJSDivergence") shouldEqual paramsMapD("maxJSDivergence")
            paramsMapI("maxCorrelation") shouldEqual paramsMapD("maxCorrelation")
            paramsMapI("correlationType") shouldEqual paramsMapD("correlationType")
            paramsMapI("jsDivergenceProtectedFeatures") shouldEqual paramsMapD("jsDivergenceProtectedFeatures")
            paramsMapI("protectedFeatures") shouldEqual paramsMapD("protectedFeatures")
        }
    }
  }

  it should "have feature insights for features that are removed by the raw feature filter" in {
    val insights = modelWithRFF.modelInsights(predWithMaps)

    modelWithRFF.getBlacklist() should contain theSameElementsAs Array(age, description, genderPL, weight)
    val heightIn = insights.features.find(_.featureName == age.name).get
    heightIn.derivedFeatures.size shouldBe 1
    heightIn.derivedFeatures.head.excluded shouldBe Some(true)

    modelWithRFF.getBlacklistMapKeys() should contain theSameElementsAs Map(numericMap.name -> Set("Female"))
    val mapDerivedIn = insights.features.find(_.featureName == numericMap.name).get.derivedFeatures
    val droppedMapDerivedIn = mapDerivedIn.filter(_.derivedFeatureName == "Female")
    mapDerivedIn.size shouldBe 3
    droppedMapDerivedIn.size shouldBe 1
    droppedMapDerivedIn.head.excluded shouldBe Some(true)
    droppedMapDerivedIn.head.derivedFeatureGroup shouldBe Some("Female")
  }

  it should "have derived feature value for map feature insights" in {
    val insights = modelWithRFF.modelInsights(predWithMaps)
    val mapDerivedIn = insights.features.find(_.featureName == numericMap.name).get.derivedFeatures
    mapDerivedIn.size shouldBe 3
    val f1InDer = mapDerivedIn.head
    println(f1InDer)
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
  }

  val labelName = "l"

  val summary = SanityCheckerSummary(
    correlations = Correlations(Seq("f1_0", "f0_f0_f2_1", "f0_f0_f3_2"), Seq(Double.NaN, 5.2, 5.3), Seq.empty,
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
        categoricalFeatures = Array("f0_f0_f3_2"),
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
    Seq("f1", "f0").map(name => name -> FeatureHistory(originFeatures = Seq(name), stages = Seq())).toMap,
    Map(
      "f0" -> Seq(SensitiveNameInformation(0.0, Seq.empty[GenderDetectionResults], 0.0, 0.0, 1.0, "f0", None))
    )
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
    val labelSum = ModelInsights.getLabelSummary(Option(lbl), Option(summary))

    val featureInsights = ModelInsights.getFeatureInsights(
      Option(meta), Option(summary), None, Array(f1, f0), Array.empty, Map.empty[String, Set[String]],
      RawFeatureFilterResults(), labelSum
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
    f0In.sensitiveInformation match {
      case Seq(SensitiveNameInformation(
        probName, genderDetectResults, probMale, probFemale, probOther, name, mapKey, actionTaken
      )) =>
        actionTaken shouldBe false
        probName shouldBe 0.0
        genderDetectResults shouldBe Seq.empty[String]
        probMale shouldBe 0.0
        probFemale shouldBe 0.0
        probOther shouldBe 1.0
      case _ => fail("SensitiveFeatureInformation was not found.")
    }

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

    insights.features.foreach(f =>
      f.distributions shouldBe wfDistributionsGrouped.getOrElse(f.featureName, Seq.empty)
    )
  }

  it should "not include raw feature distribution information when RawFeatureFilter is not used" in {
    val insights = workflowModel.modelInsights(pred)
    insights.features.foreach(f => f.distributions shouldBe empty)
  }

  it should
    """include sensitive feature information
      |even for sensitive features that are removed from output vector and output vector metadata""".stripMargin in {
    // Copy metadata from above but add new feature that was removed in vectorizing to sensitive info
    val f_notInMeta = Feature[Text]("f_notInMeta", isResponse = false, null, Seq(), "test")
    val newFeatureName = "fv"
    val newColumnMeta = OpVectorColumnMetadata(
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
    }
    val newFeatureHistory = Seq("f1", "f0").map(
      name => name -> FeatureHistory(originFeatures = Seq(name), stages = Seq())
    ).toMap
    val newSensitiveInfo = Map(
      "f0" -> Seq(SensitiveNameInformation(
        0.0, Seq.empty[GenderDetectionResults], 0.0, 0.0, 1.0, "f0", None
      )),
      "f_notInMeta" -> Seq(SensitiveNameInformation(
        1.0, Seq.empty[GenderDetectionResults], 0.0, 0.0, 1.0, "f_notInMeta", None, actionTaken = true
      ))
    )
    val newMeta = OpVectorMetadata(newFeatureName, newColumnMeta, newFeatureHistory, newSensitiveInfo)

    val labelSum = ModelInsights.getLabelSummary(Option(lbl), Option(summary))

    val featureInsights = ModelInsights.getFeatureInsights(
      Option(newMeta), Option(summary), None, Array(f1, f0, f_notInMeta), Array.empty, Map.empty[String, Set[String]],
      RawFeatureFilterResults(), labelSum
    )
    featureInsights.size shouldBe 3
    val f_notInMeta_butInInsights = featureInsights.find(_.featureName == "f_notInMeta").get
    f_notInMeta_butInInsights.featureName shouldBe "f_notInMeta"
    f_notInMeta_butInInsights.featureType shouldBe classOf[Text].getName
    f_notInMeta_butInInsights.derivedFeatures.size shouldBe 0
    f_notInMeta_butInInsights.sensitiveInformation match {
      case Seq(SensitiveNameInformation(
        probName, genderDetectResults, probMale, probFemale, probOther, _, _, actionTaken
      )) =>
        actionTaken shouldBe true
        probName shouldBe 1.0
        genderDetectResults shouldBe Seq.empty[String]
        probMale shouldBe 0.0
        probFemale shouldBe 0.0
        probOther shouldBe 1.0
      case _ => fail("SensitiveFeatureInformation was not found.")
    }
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

  val tol = 0.1
  it should "correctly return the descaled coefficient for linear regression, " +
    "when standardization is on" in {

    // Since 5000 & 1 are always returned as the coefficients of the model
    // trained on unstandardized data and we can analytically calculate
    // the scaled version of them by the linear regression formula, the coefficients
    // of the model trained on standardized data should be within a small distance of the analytical formula.

    // difference between the real coefficient and the analytical formula
    val coeffs = getFeatureImp(standardizedLinpred, unstandardizedLinpred, linRegDF._3)
    val descaledsmallCoeff = coeffs(0)
    val originalsmallCoeff = coeffs(1)
    val descaledbigCoeff = coeffs(2)
    val orginalbigCoeff = coeffs(3)
    val absError = math.abs(orginalbigCoeff * math.sqrt(smallFeatureVariance) / labelStd - descaledbigCoeff)
    val bigCoeffSum = math.abs(orginalbigCoeff * math.sqrt(smallFeatureVariance) / labelStd + descaledbigCoeff)
    val absError2 = math.abs(originalsmallCoeff * math.sqrt(bigFeatureVariance) / labelStd - descaledsmallCoeff)
    val smallCoeffSum = math.abs(originalsmallCoeff * math.sqrt(bigFeatureVariance) / labelStd + descaledsmallCoeff)
    absError should be < tol * bigCoeffSum / 2
    absError2 should be < tol * smallCoeffSum / 2
  }

  it should "correctly return the descaled coefficient for logistic regression, " +
    "when standardization is on" in {
    val coeffs = getFeatureImp(standardizedLogpred, unstandardizedLogpred, logRegDF._3)
    val descaledsmallCoeff = coeffs(0)
    val originalsmallCoeff = coeffs(1)
    val descaledbigCoeff = coeffs(2)
    val orginalbigCoeff = coeffs(3)
    // difference between the real coefficient and the analytical formula
    val absError = math.abs(orginalbigCoeff * math.sqrt(smallFeatureVariance) - descaledbigCoeff)
    val bigCoeffSum = math.abs(orginalbigCoeff * math.sqrt(smallFeatureVariance) + descaledbigCoeff)
    val absError2 = math.abs(originalsmallCoeff * math.sqrt(mediumFeatureVariance) - descaledsmallCoeff)
    val smallCoeffSum = math.abs(originalsmallCoeff * math.sqrt(mediumFeatureVariance) + descaledsmallCoeff)
    absError should be < tol * bigCoeffSum / 2
    absError2 should be < tol * smallCoeffSum / 2
  }

  it should "correctly return moments calculation and cardinality calculation for numeric features" in {

    import spark.implicits._
    val df = linRegDF._3
    val meanTol = 0.01
    val varTol = 0.01
    val (moments, cardinality) = getFeatureMomentsAndCard(standardizedLinpred, linRegDF._3)

    // Go through each feature and check that the mean, variance, and unique counts match the data
    moments.foreach { case (featureName, value) => {
      value.count shouldBe 1000
      val (expectedMean, expectedVariance) =
        df.select(avg(featureName), variance(featureName)).as[(Double, Double)].collect().head
      math.abs((value.mean - expectedMean) / expectedMean) < meanTol shouldBe true
      math.abs((value.variance - expectedVariance) / expectedVariance) < varTol shouldBe true
    }
    }

    cardinality.foreach { case (featureName, value) =>
      val actualUniques = df.select(featureName).as[Double].distinct.collect.toSet
      actualUniques should contain allElementsOf value.valueCounts.keySet.map(_.toDouble)
    }
  }

  it should "return correct insights when a model combiner equal is used as the final feature" in {
    val predComb = new SelectedModelCombiner().setCombinationStrategy(CombinationStrategy.Equal)
      .setInput(label, pred, predWithMaps).getOutput()
    val workflowModel = new OpWorkflow().setResultFeatures(pred, predComb)
      .setParameters(params).setReader(dataReader).train()
    val insights = workflowModel.modelInsights(predComb)
    insights.selectedModelInfo.nonEmpty shouldBe true
    insights.features.foreach(_.derivedFeatures.foreach(_.contribution shouldBe Seq()))
    insights.features.map(_.featureName).toSet shouldBe
      Set(genderPL, age, height, description, weight, numericMap).map(_.name)
    insights.features.foreach(_.derivedFeatures.foreach(_.contribution shouldBe Seq()))
    insights.features.foreach(_.derivedFeatures.foreach(_.variance.nonEmpty shouldBe true))
  }

  it should "return correct insights when a model combiner best is used as the final feature" in {
    val predComb = new SelectedModelCombiner().setCombinationStrategy(CombinationStrategy.Best)
      .setInput(label, pred, predWithMaps).getOutput()
    val workflowModel = new OpWorkflow().setResultFeatures(pred, predComb)
      .setParameters(params).setReader(dataReader).train()
    val predModel = workflowModel.getOriginStageOf(predComb).asInstanceOf[SelectedCombinerModel]
    val winner = if (predModel.weight1 > 0.5) pred else predWithMaps
    val insights = workflowModel.modelInsights(predComb)
    val insightsWin = workflowModel.modelInsights(winner)

    insights.selectedModelInfo.nonEmpty shouldBe true
    insights.features.map(_.featureName).toSet shouldBe insightsWin.features.map(_.featureName).toSet
    insights.features.zip(insightsWin.features).foreach{
      case (c, w) => c.derivedFeatures.zip(w.derivedFeatures)
        .foreach{ case (c1, w1) => c1.contribution shouldBe w1.contribution }
    }
  }

  it should "return default & custom metrics when having multiple binary classification metrics in model insights" in {
    val prediction = BinaryClassificationModelSelector
      .withCrossValidation(seed = 42,
        trainTestEvaluators = Seq(
          Evaluators.BinaryClassification.custom(metricName = "second", evaluateFn = _ => 0.0),
          Evaluators.BinaryClassification.custom(metricName = "third", evaluateFn = _ => 1.0)
        ),
        splitter = Option(DataSplitter(seed = 42, reserveTestFraction = 0.1)),
        modelsAndParameters = models)
      .setInput(label, checked)
      .getOutput()
    val workflow = new OpWorkflow().setResultFeatures(prediction).setParameters(params).setReader(dataReader)
    val workflowModel = workflow.train()
    val insights = workflowModel.modelInsights(prediction)
    val trainEval = insights.selectedModelInfo.get.trainEvaluation
    trainEval shouldBe a[MultiMetrics]
    val trainMetric = trainEval.asInstanceOf[MultiMetrics].metrics
    trainMetric.map { case (metricName, metric) => metricName -> metric.getClass } should contain theSameElementsAs Seq(
      OpEvaluatorNames.Binary.humanFriendlyName -> classOf[BinaryClassificationMetrics],
      OpEvaluatorNames.BinScore.humanFriendlyName -> classOf[BinaryClassificationBinMetrics],
      "second" -> classOf[SingleMetric],
      "third" -> classOf[SingleMetric]
    )
  }

  it should
    "return default & custom metrics when having multiple multi-class classification metrics in model insights" in {
    val prediction = MultiClassificationModelSelector
      .withCrossValidation(seed = 42,
        trainTestEvaluators = Seq(Evaluators.MultiClassification.custom(metricName = "second", evaluateFn = _ => 0.0)),
        splitter = Option(DataCutter(seed = 42, reserveTestFraction = 0.1)),
        modelsAndParameters = models)
      .setInput(label, checked)
      .getOutput()
    val workflow = new OpWorkflow().setResultFeatures(prediction).setParameters(params).setReader(dataReader)
    val workflowModel = workflow.train()
    val insights = workflowModel.modelInsights(prediction)
    val trainEval = insights.selectedModelInfo.get.trainEvaluation
    trainEval shouldBe a[MultiMetrics]
    val trainMetric = trainEval.asInstanceOf[MultiMetrics].metrics
    trainMetric.map { case (metricName, metric) => metricName -> metric.getClass } should contain theSameElementsAs Seq(
      OpEvaluatorNames.Multi.humanFriendlyName -> classOf[MultiClassificationMetrics],
      "second" -> classOf[SingleMetric]
    )
  }

  it should "return default & custom metrics when having multiple regression metrics in model insights" in {
    val prediction = RegressionModelSelector
      .withCrossValidation(seed = 42,
        trainTestEvaluators = Seq(Evaluators.Regression.custom(metricName = "second", evaluateFn = _ => 0.0)),
        dataSplitter = Option(DataSplitter(seed = 42, reserveTestFraction = 0.1)),
        modelsAndParameters = models)
      .setInput(label, features)
      .getOutput()
    val workflow = new OpWorkflow().setResultFeatures(prediction).setParameters(params).setReader(dataReader)
    val workflowModel = workflow.train()
    val insights = workflowModel.modelInsights(prediction)
    val trainEval = insights.selectedModelInfo.get.trainEvaluation
    trainEval shouldBe a[MultiMetrics]
    val trainMetric = trainEval.asInstanceOf[MultiMetrics].metrics
    trainMetric.map { case (metricName, metric) => metricName -> metric.getClass } should contain theSameElementsAs Seq(
      OpEvaluatorNames.Regression.humanFriendlyName -> classOf[RegressionMetrics],
      "second" -> classOf[SingleMetric]
    )
  }

}

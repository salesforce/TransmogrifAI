package com.salesforce.op.stages.impl.selector

import com.salesforce.op.OpWorkflow
import com.salesforce.op.evaluators.OpBinScoreEvaluator
import com.salesforce.op.features.{Feature, FeatureBuilder}
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.impl.PredictionEquality
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelSelector, OpLogisticRegression, OpRandomForestClassifier}
import com.salesforce.op.test.OpEstimatorSpec
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.sql.Dataset
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class SelectedCombinerTest extends OpEstimatorSpec[Prediction, SelectedCombinerModel, SelectedCombiner]
  with PredictionEquality {

  val (seed, smallCount, bigCount) = (1234L, 20, 80)

  import spark.implicits._

  // Generate positive observations following a distribution ~ N((0.0, 0.0, 0.0), I_3)
  val positiveData =
    normalVectorRDD(sc, bigCount, 3, seed = seed)
      .map(v => 1.0 -> Vectors.dense(v.toArray))

  // Generate negative observations following a distribution ~ N((10.0, 10.0, 10.0), I_3)
  val negativeData =
    normalVectorRDD(sc, smallCount, 3, seed = seed)
      .map(v => 0.0 -> Vectors.dense(v.toArray.map(_ + 10.0)))

  val data = positiveData.union(negativeData).toDF("label", "features")

  val (label, Array(features: Feature[OPVector]@unchecked)) = FeatureBuilder.fromDataFrame[RealNN](
    data, response = "label", nonNullable = Set("features")
  )

  val lr = new OpLogisticRegression()
  val lrParams = new ParamGridBuilder()
    .addGrid(lr.regParam, DefaultSelectorParams.Regularization)
    .build()

  val rf = new OpRandomForestClassifier()
  val rfParams = new ParamGridBuilder()
    .addGrid(rf.maxDepth, Array(2))
    .addGrid(rf.minInfoGain, Array(10.0))
    .build()

  val ms1 = BinaryClassificationModelSelector
    .withCrossValidation(modelsAndParameters = Seq(lr -> lrParams),
      trainTestEvaluators = Seq(new OpBinScoreEvaluator()))
    .setInput(label, features)
    .getOutput()

  val ms2 = BinaryClassificationModelSelector
    .withCrossValidation(modelsAndParameters = Seq(rf -> rfParams))
    .setInput(label, features)
    .getOutput()


  override val inputData: Dataset[_] = new OpWorkflow()
    .setResultFeatures(ms1, ms2)
    .transform(data)

  override val estimator: SelectedCombiner = new SelectedCombiner().setInput(label, ms1, ms2)

  override val expectedResult: Seq[Prediction] = inputData.collect(ms1)

  it should "have the correct metadata for the best model" in {
    ModelSelectorSummary.fromMetadata(estimator.fit(inputData).getMetadata().getSummaryMetadata()) shouldBe
      ModelSelectorSummary.fromMetadata(inputData.schema(ms1.name).metadata.getSummaryMetadata())
  }

  it should "combine model results correctly" in {
    val model = estimator.setCombinationStrategy(CombinationStrategy.Weighted).fit(inputData)
      .asInstanceOf[SelectedCombinerModel]
    val outfeature = model.getOutput()
    val outdata = model.transform(inputData)
    outdata.collect(outfeature).map(_.probability(0)) shouldEqual inputData.collect(ms1, ms2)
      .map{ case (p1, p2) => p1.probability(0) * model.weight1 + p2.probability(0) * model.weight2}
    val meta = ModelSelectorSummary.fromMetadata(outdata.schema(outfeature.name).metadata.getSummaryMetadata())
    val meta1 = ModelSelectorSummary.fromMetadata(inputData.schema(ms1.name).metadata.getSummaryMetadata())
    val meta2 = ModelSelectorSummary.fromMetadata(inputData.schema(ms2.name).metadata.getSummaryMetadata())
    meta.bestModelUID shouldBe meta1.bestModelUID + " " + meta2.bestModelUID
    meta.trainEvaluation == meta1.trainEvaluation shouldBe false
    meta.trainEvaluation == meta2.trainEvaluation shouldBe false
    meta.trainEvaluation.toMap.keySet shouldBe meta1.trainEvaluation.toMap.keySet
      .union(meta2.trainEvaluation.toMap.keySet)
  }
}

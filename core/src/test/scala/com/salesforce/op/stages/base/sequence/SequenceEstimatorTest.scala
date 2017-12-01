/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.sequence

import com.salesforce.op.UID
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.SequenceAggregators
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SequenceEstimatorTest
  extends FlatSpec with TestSparkContext {

  val data = Seq[(DateList, DateList, DateList)](
    (new DateList(1476726419000L, 1476726019000L),
      new DateList(1476726919000L),
      new DateList(1476726519000L)),
    (new DateList(1476725419000L, 1476726019000L),
      new DateList(1476726319000L, 1476726919000L),
      new DateList(1476726419000L)),
    (new DateList(1476727419000L),
      new DateList(1476728919000L),
      new DateList(1476726619000L, 1476726949000L))
  )
  val (ds, clicks, opens, purchases) = TestFeatureBuilder("clicks", "opens", "purchases", data)

  val testEstimator: SequenceEstimator[DateList, OPVector] = new FractionOfResponsesEstimator()

  Spec[SequenceEstimator[_, _]] should "throw an error if you try to get the output without setting the inputs" in {
    intercept[java.util.NoSuchElementException](testEstimator.getOutput())
  }

  it should "return a single output feature of the correct type" in {
    val outputFeatures = testEstimator.setInput(clicks, opens, purchases).getOutput()
    outputFeatures shouldBe new Feature[OPVector](
      name = testEstimator.outputName,
      originStage = testEstimator,
      isResponse = false,
      parents = Array(clicks, opens, purchases)
    )
  }

  it should "return a SequenceModel with the estimator as the parent and the correct function" in {
    val testModel = testEstimator.setInput(clicks, opens, purchases).fit(ds)
    testModel.parent shouldBe testEstimator
    testModel.transformFn(
      Seq(new DateList(1476726419000L), new DateList(1476726419000L), new DateList(1476726419000L))
    ) shouldEqual Vectors.dense(0.2, 0.25, 0.25).toOPVector
  }

  it should "create a SequenceModel that uses the specified transform function when fit" in {
    val testModel = testEstimator.setInput(clicks, opens, purchases).fit(ds)
    val testDataTransformed = testModel.setInput(clicks, opens, purchases).transform(ds)
    val transformedValues = testDataTransformed.collect(clicks, opens, purchases, testModel.getOutput())

    // This is string because of vector type being private to spark ml
    testDataTransformed.schema.fieldNames.toSet shouldEqual
      Set(clicks.name, opens.name, purchases.name, testEstimator.outputName)

    val fractions = Array(
      Vectors.dense(0.4, 0.25, 0.25).toOPVector,
      Vectors.dense(0.4, 0.5, 0.25).toOPVector,
      Vectors.dense(0.2, 0.25, 0.5).toOPVector
    )
    val expected = data.zip(fractions) .map { case ((d1, d2, d3), f) => (d1, d2, d3, f)}

    transformedValues shouldBe expected
  }

  it should "copy itself and the model successfully" in {
    val est = new FractionOfResponsesEstimator()
    val mod = new FractionOfResponsesModel(Seq.empty, est.operationName, est.uid)

    est.copy(new ParamMap()).uid shouldBe est.uid
    mod.copy(new ParamMap()).uid shouldBe mod.uid
  }
}

class FractionOfResponsesEstimator(uid: String = UID[FractionOfResponsesEstimator])
  extends SequenceEstimator[DateList, OPVector](operationName = "fractionOfResponses", uid = uid) {
  def fitFn(dataset: Dataset[Seq[Seq[Long]]]): SequenceModel[DateList, OPVector] = {
    import dataset.sparkSession.implicits._
    val sizes = dataset.map(_.map(_.size))
    val size = sizes.first().size
    val counts = sizes.select(SequenceAggregators.SumNumSeq[Int](size = size)).first()
    new FractionOfResponsesModel(counts = counts, operationName = operationName, uid = uid)
  }
}

private final class FractionOfResponsesModel
(
  val counts: Seq[Int],
  operationName: String,
  uid: String
) extends SequenceModel[DateList, OPVector](operationName = operationName, uid = uid) {
  def transformFn: (Seq[DateList]) => OPVector = row => {
    val fractions = row.zip(counts).map { case (feature, count) => feature.value.size.toDouble / count }
    Vectors.dense(fractions.toArray).toOPVector
  }
}


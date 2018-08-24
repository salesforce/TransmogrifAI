package com.salesforce.op.stages.base.sequence

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.SequenceAggregators
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class BinarySequenceEstimatorTest
  extends OpEstimatorSpec[
    OPVector,
    BinarySequenceModel[Real, DateList, OPVector],
    BinarySequenceEstimator[Real, DateList, OPVector]] {

  val sample = Seq[(DateList, DateList, DateList, Real)](
    (new DateList(1476726419000L, 1476726019000L),
      new DateList(1476726919000L),
      new DateList(1476726519000L),
      Real(1.0)),
    (new DateList(1476725419000L, 1476726019000L),
      new DateList(1476726319000L, 1476726919000L),
      new DateList(1476726419000L),
      Real(0.5)),
    (new DateList(1476727419000L),
      new DateList(1476728919000L),
      new DateList(1476726619000L, 1476726949000L),
      Real(0.0))
  )
  val (inputData, clicks, opens, purchases, weights) =
    TestFeatureBuilder("clicks", "opens", "purchases", "weights", sample)

  val estimator = new WeightedFractionOfResponsesEstimator().setInput(weights, clicks, opens, purchases)

  val expectedResult = Seq(
    Vectors.dense(0.4, 0.5, Double.PositiveInfinity).toOPVector,
    Vectors.dense(0.4, 1.0, Double.PositiveInfinity).toOPVector,
    Vectors.dense(0.2, 0.5, Double.PositiveInfinity).toOPVector
  )
}


class WeightedFractionOfResponsesEstimator(uid: String = UID[WeightedFractionOfResponsesEstimator])
  extends BinarySequenceEstimator[Real, DateList, OPVector](operationName = "fractionOfResponses", uid = uid) {
  def fitFn(dataset: Dataset[(Real#Value, Seq[Seq[Long]])]): BinarySequenceModel[Real, DateList, OPVector] = {
    import dataset.sparkSession.implicits._
    val sizes = dataset.map(_._2.map(_.size))
    val weights = dataset.map(_._1.get).rdd.collect()
    val size = getInputFeatures().length
    val counts = sizes.select(SequenceAggregators.SumNumSeq[Int](size = size).toColumn).first()
    val weightedCounts = counts.zip(weights).map {
      case (c, w) => c.toDouble * w
    }
    new WeightedFractionOfResponsesModel(counts = weightedCounts, operationName = operationName, uid = uid)
  }
}

final class WeightedFractionOfResponsesModel private[op]
(
  val counts: Seq[Double],
  operationName: String,
  uid: String
) extends BinarySequenceModel[Real, DateList, OPVector](operationName = operationName, uid = uid) {
  def transformFn: (Real, Seq[DateList]) => OPVector = (w, dates) => {
    val fractions = dates.zip(counts).map { case (feature, count) => feature.value.size.toDouble / count }
    Vectors.dense(fractions.toArray).toOPVector
  }
}

/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.ternary

import breeze.numerics.abs
import com.salesforce.op.UID
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TernaryEstimatorTest extends FlatSpec with PassengerSparkFixtureTest {

  val testEstimator: TernaryEstimator[MultiPickList, Binary, RealMap, Real] = new TripleInteractionsEstimator()

  Spec[TernaryEstimator[_, _, _, _]] should "error if you try to get the output without setting the inputs" in {
    intercept[java.util.NoSuchElementException](testEstimator.getOutput())
  }

  it should "return a single output feature of the correct type" in {
    val outputFeatures = testEstimator.setInput(gender, survived, numericMap).getOutput()
    outputFeatures shouldBe new Feature[Real](
      name = testEstimator.outputName,
      originStage = testEstimator,
      isResponse = true,
      parents = Array(gender, survived, numericMap)
    )
  }

  it should "return a TernaryModel with the estimator as the parent and the correct function" in {
    val testModel = testEstimator.setInput(gender, survived, numericMap).fit(passengersDataSet)

    testModel.parent shouldBe testEstimator
    abs(
      testModel.transformFn(Seq("male").toMultiPickList, false.toBinary, Map("male" -> 1.2).toRealMap).value.get - 0.0
    ) should be < 0.000000002

    testModel.transformFn(Seq("male").toMultiPickList, true.toBinary, Map("male" -> 1.2).toRealMap).value shouldBe None
    abs(
      testModel.transformFn(Seq("male").toMultiPickList, false.toBinary, Map("male" -> 2.2).toRealMap).value.get - 1.0
    ) should be < 0.000000002
  }

  it should "create a TernaryModel that uses the specified transform function when fit" in {
    val testModel = testEstimator.setInput(gender, survived, numericMap).fit(passengersDataSet)
    val testDataTransformed = testModel.setInput(gender, survived, numericMap)
      .transform(passengersDataSet.select(gender.name, survived.name, numericMap.name))

    testDataTransformed.schema shouldEqual StructType(
      Seq(StructField(gender.name, ArrayType(StringType, true), true),
        StructField(survived.name, BooleanType, true),
        StructField(numericMap.name, MapType(StringType, DoubleType, true), true),
        StructField(testEstimator.outputName, DoubleType, true)))

    testDataTransformed.collect(gender, survived, numericMap, testModel.getOutput()) shouldEqual Array(
      (Set("Male").toMultiPickList, false.toBinary, new RealMap(Map("Male" -> 2.0)), 0.8.toReal),
      (Seq().toMultiPickList, true.toBinary, new RealMap(Map()), new Real(None)),
      (Set("Female").toMultiPickList, new Binary(None), Map("Female" -> 1.0).toRealMap, new Real(-0.19999999999999996)),
      (Set("Female").toMultiPickList, new Binary(None), Map("Female" -> 1.0).toRealMap, new Real(-0.19999999999999996)),
      (Set("Male").toMultiPickList, new Binary(None), Map("Male" -> 1.0).toRealMap, new Real(-0.19999999999999996)),
      (Set("Female").toMultiPickList, new Binary(None), Map("Female" -> 1.0).toRealMap, new Real(-0.19999999999999996))
    )
  }

  it should "copy itself and the model successfully" in {
    val est = new TripleInteractionsEstimator()
    val mod = new TripleInteractionsModel(0.0, est.operationName, est.uid)

    est.copy(new ParamMap()).uid shouldBe est.uid
    mod.copy(new ParamMap()).uid shouldBe mod.uid
  }

}


class TripleInteractionsEstimator(uid: String = UID[TripleInteractionsEstimator])
  extends TernaryEstimator[MultiPickList, Binary, RealMap, Real](operationName = "tripleInteractions", uid = uid)
    with TripleInteractions {

  // scalastyle:off line.size.limit
  def fitFn(dataset: Dataset[(MultiPickList#Value, Binary#Value, RealMap#Value)]): TernaryModel[MultiPickList, Binary, RealMap, Real] = {
    import dataset.sparkSession.implicits._
    val mean = {
      dataset.map { case (gndr, srvvd, nmrcMp) =>
        if (survivedAndMatches(gndr, srvvd, nmrcMp)) nmrcMp(gndr.head) else 0.0
      }.filter(_ != 0.0).groupBy().mean().first().getDouble(0)
    }
    new TripleInteractionsModel(mean = mean, operationName = operationName, uid = uid)
  }
  // scalastyle:on

}

final class TripleInteractionsModel private[op](val mean: Double, operationName: String, uid: String)
  extends TernaryModel[MultiPickList, Binary, RealMap, Real](operationName = operationName, uid = uid)
    with TripleInteractions {

  def transformFn: (MultiPickList, Binary, RealMap) => Real = (g: MultiPickList, s: Binary, nm: RealMap) => new Real(
    if (!survivedAndMatches(g.value, s.value, nm.value)) None
    else Some(nm.value(g.value.head) - mean)
  )

}

sealed trait TripleInteractions {
  def survivedAndMatches(g: MultiPickList#Value, s: Binary#Value, nm: RealMap#Value): Boolean =
    !s.getOrElse(false) && g.nonEmpty && nm.contains(g.head)
}

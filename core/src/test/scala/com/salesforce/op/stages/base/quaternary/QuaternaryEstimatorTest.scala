/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.quaternary

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
class QuaternaryEstimatorTest extends FlatSpec with PassengerSparkFixtureTest {

  var testEstimator: QuaternaryEstimator[Real, TextMap, BinaryMap, MultiPickList, Real] = new FantasticFourEstimator()

  Spec[QuaternaryEstimator[_, _, _, _, _]] should
    "throw an error if you try to get the output without setting the inputs" in {
    intercept[java.util.NoSuchElementException](testEstimator.getOutput())
  }

  it should "return a single output feature of the correct type" in {
    val outputFeatures = testEstimator.setInput(age, stringMap, booleanMap, gender).getOutput()
    outputFeatures shouldBe new Feature[Real](
      name = testEstimator.getOutputFeatureName,
      originStage = testEstimator,
      isResponse = false,
      parents = Array(age, stringMap, booleanMap, gender)
    )

  }

  it should "create a TernaryModel that uses the specified transform function when fit" in {
    val testModel = testEstimator.setInput(age, stringMap, booleanMap, gender).fit(passengersDataSet)
    val testDataTransformed = testModel.setInput(age, stringMap, booleanMap, gender)
      .transform(passengersDataSet.select(age.name, stringMap.name, booleanMap.name, gender.name))

    testDataTransformed.schema shouldEqual StructType(
      Seq(StructField(age.name, DoubleType, true),
        StructField(stringMap.name, MapType(StringType, StringType, true), true),
        StructField(booleanMap.name, MapType(StringType, BooleanType, true), true),
        StructField(gender.name, ArrayType(StringType, true), true),
        StructField(testEstimator.getOutputFeatureName, DoubleType, true)
      )
    )

    val expected = Array(
      Real(13.833333333333336),
      Real(None),
      Real(-3.1666666666666643),
      Real(-34.166666666666664),
      Real(None),
      Real(-4.166666666666664)
    )

    testDataTransformed.collect(testModel.getOutput()) shouldEqual expected
  }

  it should "copy itself and the model successfully" in {
    val est = new FantasticFourEstimator()
    val mod = new FantasticFourModel(0.0, est.operationName, est.uid)

    est.copy(new ParamMap()).uid shouldBe est.uid
    mod.copy(new ParamMap()).uid shouldBe mod.uid
  }

}


class FantasticFourEstimator(uid: String = UID[FantasticFourEstimator])
  extends QuaternaryEstimator[Real, TextMap, BinaryMap, MultiPickList, Real](operationName = "fantasticFour", uid = uid)
    with FantasticFour  {

  // scalastyle:off line.size.limit
  def fitFn(dataset: Dataset[(Real#Value, TextMap#Value, BinaryMap#Value, MultiPickList#Value)]): QuaternaryModel[Real, TextMap, BinaryMap, MultiPickList, Real] = {
    import dataset.sparkSession.implicits._
    val topAge = dataset.map(_._1.getOrElse(0.0)).groupBy().max().first().getDouble(0)
    val mean = dataset.map { case (age, strMp, binMp, gndr) =>
      if (filterFN(age, strMp, binMp, gndr)) age.getOrElse(topAge) else topAge
    }.groupBy().mean().first().getDouble(0)

    new FantasticFourModel(mean = mean, operationName = operationName, uid = uid)
  }
  // scalastyle:on

}

final class FantasticFourModel private[op](val mean: Double, operationName: String, uid: String)
  extends QuaternaryModel[Real, TextMap, BinaryMap, MultiPickList, Real](operationName = operationName, uid = uid)
    with FantasticFour {

  def transformFn: (Real, TextMap, BinaryMap, MultiPickList) => Real = (age, strMp, binMp, gndr) => new Real(
    if (filterFN(age.v, strMp.v, binMp.v, gndr.v)) Some(age.v.get - mean) else None
  )

}

sealed trait FantasticFour {
  def filterFN(a: Real#Value, sm: TextMap#Value, bm: BinaryMap#Value, g: MultiPickList#Value): Boolean =
    a.nonEmpty && g.nonEmpty && sm.contains(g.head) && bm.contains(g.head)
}


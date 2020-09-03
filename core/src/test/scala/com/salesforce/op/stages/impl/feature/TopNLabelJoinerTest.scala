package com.salesforce.op.stages.impl.feature

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.features.types._
import com.salesforce.op.utils.spark.RichDataset._

@RunWith(classOf[JUnitRunner])
class TopNLabelJoinerTest extends MultiLabelJoinerBaseTest[TopNLabelJoiner] {
  val transformer = new TopNLabelJoiner(topN = 2).setInput(classIndexFeature, probVecFeature)

  val expectedResult = Seq(
    Map(classes(0) -> 40.0, classes(1) -> 30.0).toRealMap,
    Map(classes(1) -> 40.0, classes(2) -> 30.0).toRealMap,
    Map(classes(2) -> 40.0, classes(0) -> 30.0).toRealMap
  )

  it should "return top 4" in {
    val stage = new TopNLabelJoiner(topN = 4).setInput(classIndexFeature, probVecFeature)
    val actual = stage.transform(inputData).collect(idFeature, stage.getOutput())

    val expected = Array(
      (1001.toIntegral, Map(classes(0) -> 40.0, classes(1) -> 30.0, classes(2) -> 20.0).toRealMap),
      (1002.toIntegral, Map(classes(1) -> 40.0, classes(2) -> 30.0, classes(0) -> 20.0).toRealMap),
      (1003.toIntegral, Map(classes(2) -> 40.0, classes(0) -> 30.0, classes(1) -> 20.0).toRealMap)
    )
    actual shouldBe expected
  }

}


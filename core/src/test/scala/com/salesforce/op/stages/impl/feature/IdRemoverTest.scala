package com.salesforce.op.stages.impl.feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class IdRemoverTest extends OpTransformerSpec[Text, IdRemover] {
  val sample = Seq(Text("ball"), Text("stray"), Text("happy"),
    Text("express"), Text("achieve"), Text("swell"), Text("frame"))
  val (inputData, f1) = TestFeatureBuilder(sample)
  val transformer: IdRemover = new IdRemover(10).setInput(f1)
  override val expectedResult: Seq[Text] = Seq(Text.empty, Text.empty, Text.empty,
    Text.empty, Text.empty, Text.empty, Text.empty)
}


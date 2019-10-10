package com.salesforce.op.stages.impl.feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class IdRemoverTest extends OpTransformerSpec[Text, IdRemover] {
  val sample = Seq(Text("ball"), Text("stray"), Text("happy"), Text("express"), Text("achieve"), Text("swell"), Text("frame"))
  val (inputData, f1) = TestFeatureBuilder(sample)
  val transformer: IdRemover = new IdRemover(10).setInput(f1)
  override val expectedResult: Seq[Integral] = Seq(Integral(-1), Integral(-4), Integral.empty, Integral(6),
    Integral(-5), Integral(1), Integral(3), Integral(1))
}


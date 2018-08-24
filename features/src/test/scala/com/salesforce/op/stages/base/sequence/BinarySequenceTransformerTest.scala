package com.salesforce.op.stages.base.sequence

import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class BinarySequenceTransformerTest extends OpTransformerSpec[
  MultiPickList, BinarySequenceTransformer[Real, Text, MultiPickList]] {

  val sample = Seq(
    (1.toReal, "one".toText, "two".toText),
    ((-1).toReal, "three".toText, "four".toText),
    (15.toReal, "five".toText, "six".toText),
    (1.111.toReal, "seven".toText, "".toText)
  )

  val (inputData, f1, f2, f3) = TestFeatureBuilder(sample)

  val transformer = new BinarySequenceLambdaTransformer[Real, Text, MultiPickList](
    operationName = "realToMultiPicklist",
    transformFn = (r, texts) => MultiPickList(texts.map(_.value.get).toSet + r.value.get.toString)
  ).setInput(f1, f2, f3)

  val expectedResult = Seq(
    Set("1.0", "one", "two").toMultiPickList,
    Set("-1.0", "three", "four").toMultiPickList,
    Set("15.0", "five", "six").toMultiPickList,
    Set("1.111", "seven", "").toMultiPickList
  )
}


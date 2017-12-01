/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class ToOccurTransformerTest extends FlatSpec with TestSparkContext {

  val testData = Seq(
    ("001", 0, None, Option(true), None),
    ("002", 1, None, None, Option(2.0)),
    ("003", 2, Option("abc"), Option(false), Option(0.0)),
    ("004", 0, Option("def"), Option(false), Option(1.0))
  ).map { case (leadId, numEmails, opptyId, doNotContact, numFormSubmits) =>
    (
      Text(leadId),
      numEmails.toRealNN,
      Text(opptyId),
      Binary(doNotContact),
      Real(numFormSubmits)
    )
  }

  lazy val (ds, leadId, numEmails, opptyId, doNotContact, numFormSubmits) =
    TestFeatureBuilder("leadId", "numEmails", "opptyId", "doNotContact", "numFormSubmits", testData)

  Spec[ToOccurTransformer[_]] should "convert features to doolean using shortcuts" in {
    val occurEmailOutput = numEmails.occurs(_.value.exists(_ > 1))
    val toOccurEmail = occurEmailOutput.originStage.asInstanceOf[Transformer]
    val occurFormOutput = numFormSubmits.occurs()
    val toOccurForm = occurFormOutput.originStage.asInstanceOf[Transformer]
    val occurContactOutput = doNotContact.occurs()
    val toOccurContact = occurContactOutput.originStage.asInstanceOf[Transformer]

    val toOccurDF = toOccurContact.transform(toOccurForm.transform(toOccurEmail.transform(ds)))

    val expected = Array(
      (Text("001"), 0.0.toRealNN, 0.0.toRealNN, 1.0.toRealNN),
      (Text("002"), 0.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN),
      (Text("003"), 1.0.toRealNN, 0.0.toRealNN, 0.0.toRealNN),
      (Text("004"), 0.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN)
    )

    toOccurDF.orderBy("leadId").collect(leadId, occurEmailOutput, occurFormOutput, occurContactOutput) shouldBe expected
  }

  it should "convert features to doolean" in {
    val toOccurEmail = new ToOccurTransformer[RealNN]().setInput(numEmails)
    val occurEmailOutput = toOccurEmail.getOutput()
    val toOccurForm = new ToOccurTransformer[Real]().setInput(numFormSubmits)
    val occurFormOutput = toOccurForm.getOutput()
    val toOccurContact = new ToOccurTransformer[Binary]().setInput(doNotContact)
    val occurContactOutput = toOccurContact.getOutput()
    val toOccurOppty = new ToOccurTransformer[Text](matchFn = _.nonEmpty).setInput(opptyId)
    val occurOpptyOutput = toOccurOppty.getOutput()

    val toOccurDF = toOccurOppty.transform(toOccurContact.transform(toOccurForm.transform(toOccurEmail.transform(ds))))

    val expected = Array(
      (Text("001"), 0.0.toRealNN, 0.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN),
      (Text("002"), 1.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN, 0.0.toRealNN),
      (Text("003"), 1.0.toRealNN, 0.0.toRealNN, 0.0.toRealNN, 1.0.toRealNN),
      (Text("004"), 0.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN, 1.0.toRealNN)
    )

    toOccurDF.orderBy("leadId").collect(leadId,
      occurEmailOutput, occurFormOutput, occurContactOutput, occurOpptyOutput) shouldBe expected
  }
}

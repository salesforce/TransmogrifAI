/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.test.{TestFeatureBuilder, _}
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class EmailParserTest extends FlatSpec with TestCommon with TestSparkContext {

  val (df, email) = TestFeatureBuilder("email", Seq(
    Email("test@example.com"),
    Email("@example.com"),
    Email("test@"),
    Email("@"),
    Email(""),
    Email("notanemail"),
    Email.empty,
    Email("first.last@example.com")
  ))

  "Email Extraction" should "extract prefix from simple email addresses" in {
    val prefix = email.toEmailPrefix
    val result = prefix.originStage.asInstanceOf[UnaryLambdaTransformer[Email, Text]].transform(df)

    result.collect(prefix) should contain theSameElementsInOrderAs
      Seq(Text("test"), Text.empty, Text.empty, Text.empty, Text.empty, Text.empty, Text.empty, Text("first.last"))
  }

  it should "extract domain from simple email addresses" in {
    val domain = email.toEmailDomain
    val result = domain.originStage.asInstanceOf[UnaryLambdaTransformer[Email, Text]].transform(df)

    result.collect(domain) should contain theSameElementsInOrderAs
      Seq(Text("example.com"), Text.empty, Text.empty, Text.empty, Text.empty, Text.empty, Text.empty,
        Text("example.com"))
  }
}

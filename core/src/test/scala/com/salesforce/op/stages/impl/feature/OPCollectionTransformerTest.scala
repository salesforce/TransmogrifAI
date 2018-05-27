/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OPCollectionTransformerTest extends FlatSpec with TestSparkContext {

  lazy val (dataEmailMap, top) = TestFeatureBuilder("name",
    Seq(
      Map("p1" -> "Kevin@gmail.com", "p2" -> "Todd@hotmail.com"),
      Map("p1" -> "Ellie@cc.net"),
      Map("p1" -> "Dave@facebook.com"),
      Map("p1" -> "Dwayne@wwf.org", "p2" -> "Darcy@yahoo.co.uk")
    ).map(EmailMap(_))
  )

  lazy val (dataAllEmpty, _) = TestFeatureBuilder(top.name,
    Seq(TextMap.empty, TextMap.empty, TextMap.empty)
  )

  lazy val (dataTextList, f1) = TestFeatureBuilder(top.name,
    Seq(
      TextList(Seq("I", "have", "some", "coconuts")),
      TextList.empty,
      TextList(Seq("and", "I", "cannot", "lie"))
    )
  )

  Spec[OPCollectionTransformer[_, _, _, _]] should "be able to turn an EmailMap into an IntegralMap" in {
    val myBaseTransformer = new UnaryLambdaTransformer[Email, Integral](
      operationName = "testUnary",
      transformFn = (input: Email) => input.value.map(_.length).toIntegral
    )

    val mapWrap = new OPMapTransformer[Email, Integral, EmailMap, IntegralMap](
      transformer = myBaseTransformer,
      operationName = "testUnaryMapWrap").setInput(top)
    val transformed = mapWrap.transform(dataEmailMap)
    val output = mapWrap.getOutput()
    val actualOutput = transformed.collect(output)
    val expectedOutput = Array(
      IntegralMap(Map("p1" -> 15, "p2" -> 16)),
      IntegralMap(Map("p1" -> 12)),
      IntegralMap(Map("p1" -> 17)),
      IntegralMap(Map("p1" -> 14, "p2" -> 17))
    )

    actualOutput shouldEqual expectedOutput
  }

  it should "throw an error in incompatible types" in {
    val t = new UnaryLambdaTransformer[Email, Real]("testUnary", transformFn = _.value.map(_.length).toReal)
    the[IllegalArgumentException] thrownBy {
      new OPMapTransformer[Email, Real, EmailMap, IntegralMap](t, operationName = "map")
    }
    the[IllegalArgumentException] thrownBy {
      new OPListTransformer[Email, Real, DateTimeList, DateTimeList](t, operationName = "list")
    }
    the[IllegalArgumentException] thrownBy {
      new OPSetTransformer[Email, Real, MultiPickList, MultiPickList](t, operationName = "set")
    }
  }

  it should "be able to turn an EmailMap into an IntegralMap even if the supplied" +
    "UnaryTransformer returns Nones" in {
    val myBaseTransformer = new UnaryLambdaTransformer[Email, Integral](
      operationName = "testUnary",
      transformFn = (input: Email) => Integral(None)
    )

    val mapWrap = new OPMapTransformer[Email, Integral, EmailMap, IntegralMap](
      transformer = myBaseTransformer,
      operationName = "testUnaryMapWrap").setInput(top)
    val transformed = mapWrap.transform(dataEmailMap)
    val output = mapWrap.getOutput()
    val actualOutput = transformed.collect(output)
    // Have to be sneaky here with an asInstanceOf[...] in order to put nulls in the values of these maps
    val expectedOutput = Array(
      IntegralMap(Map("p1" -> null, "p2" -> null).asInstanceOf[Map[String, Long]]),
      IntegralMap(Map("p1" -> null).asInstanceOf[Map[String, Long]]),
      IntegralMap(Map("p1" -> null).asInstanceOf[Map[String, Long]]),
      IntegralMap(Map("p1" -> null, "p2" -> null).asInstanceOf[Map[String, Long]])
    )

    actualOutput shouldEqual expectedOutput
  }

  it should "be able to turn a TextList into another TextList" in {
    val myBaseTransformer = new UnaryLambdaTransformer[Text, Text](
      operationName = "testUnary",
      transformFn = (input: Text) => input.value.map(_.toUpperCase).toText
    )

    val mapWrap = new OPListTransformer[Text, Text, TextList, TextList](
      transformer = myBaseTransformer,
      operationName = "testUnaryMapWrap").setInput(f1)
    val transformed = mapWrap.transform(dataTextList)
    val output = mapWrap.getOutput()
    val actualOutput = transformed.collect(output)
    val expectedOutput = Array(
      TextList(Seq("I", "HAVE", "SOME", "COCONUTS")),
      TextList.empty,
      TextList(Seq("AND", "I", "CANNOT", "LIE"))
    )

    actualOutput shouldEqual expectedOutput
  }

  it should "be able to turn a MultiPickList into another one, given different unary transformers" in {
    val unaryTextText = new UnaryLambdaTransformer[Text, Text](
      operationName = "unaryTextText",
      transformFn = (input: Text) => input.value.map(_.toUpperCase).toText
    )
    val unaryEmailText = new UnaryLambdaTransformer[Email, Text](
      operationName = "unaryEmailText",
      transformFn = (input: Email) => input.value.map(_.toUpperCase).toText
    )
    val unaryTextEmail = new UnaryLambdaTransformer[Text, Email](
      operationName = "unaryEmailText",
      transformFn = (input: Text) => input.value.map(_.toLowerCase).toEmail
    )
    val unaryURLPicklist = new UnaryLambdaTransformer[URL, PickList](
      operationName = "unaryURLPicklist",
      transformFn = (input: URL) => input.value.map(_.toLowerCase).toPickList
    )

    val (testData, multiPicklistCol) = TestFeatureBuilder("multiPickList",
      Seq(
        Set("b", "C", "G").toMultiPickList,
        Set("A", "f", "G").toMultiPickList,
        Set("A", "C", "y").toMultiPickList,
        Set("d", "w", "q").toMultiPickList
      )
    )

    val colTransformer1 = new OPSetTransformer[Text, Text, MultiPickList, MultiPickList](
      transformer = unaryTextText,
      operationName = "unaryTextTextWrap").setInput(multiPicklistCol)
    val transformed1 = colTransformer1.transform(testData)
    val output1 = colTransformer1.getOutput()
    val actualOutput1 = transformed1.collect(output1)
    val expectedOutput1 = Array(
      MultiPickList(Set("B", "C", "G")),
      MultiPickList(Set("A", "F", "G")),
      MultiPickList(Set("A", "C", "Y")),
      MultiPickList(Set("D", "W", "Q"))
    )

    val colTransformer2 = new OPSetTransformer[Email, Text, MultiPickList, MultiPickList](
      transformer = unaryEmailText,
      operationName = "unaryEmailTextWrap").setInput(multiPicklistCol)
    val transformed2 = colTransformer2.transform(testData)
    val output2 = colTransformer2.getOutput()
    val actualOutput2 = transformed2.collect(output2)
    val expectedOutput2 = Array(
      MultiPickList(Set("B", "C", "G")),
      MultiPickList(Set("A", "F", "G")),
      MultiPickList(Set("A", "C", "Y")),
      MultiPickList(Set("D", "W", "Q"))
    )

    val colTransformer3 = new OPSetTransformer[Text, Email, MultiPickList, MultiPickList](
      transformer = unaryTextEmail,
      operationName = "unaryTextEmailWrap").setInput(multiPicklistCol)
    val transformed3 = colTransformer3.transform(testData)
    val output3 = colTransformer3.getOutput()
    val actualOutput3 = transformed3.collect(output3)
    val expectedOutput3 = Array(
      MultiPickList(Set("b", "c", "g")),
      MultiPickList(Set("a", "f", "g")),
      MultiPickList(Set("a", "c", "y")),
      MultiPickList(Set("d", "w", "q"))
    )

    val colTransformer4 = new OPSetTransformer[URL, PickList, MultiPickList, MultiPickList](
      transformer = unaryURLPicklist,
      operationName = "unaryURLPicklistWrap").setInput(multiPicklistCol)
    val transformed4 = colTransformer4.transform(testData)
    val output4 = colTransformer4.getOutput()
    val actualOutput4 = transformed4.collect(output4)
    val expectedOutput4 = Array(
      MultiPickList(Set("b", "c", "g")),
      MultiPickList(Set("a", "f", "g")),
      MultiPickList(Set("a", "c", "y")),
      MultiPickList(Set("d", "w", "q"))
    )

    actualOutput1 shouldEqual expectedOutput1
    actualOutput2 shouldEqual expectedOutput2
    actualOutput3 shouldEqual expectedOutput3
    actualOutput4 shouldEqual expectedOutput4
  }

  // Unfortunately, it looks like we can't catch these types of errors at compile time
  it should "fail to create a unary MultiPickList transformer if the supplied unary transformer does not map from" +
    "a Text to Text type" in {

    val unaryTextIntegral = new UnaryLambdaTransformer[Text, Integral](
      operationName = "unaryTextText",
      transformFn = (input: Text) => Integral(input.hashCode)
    )

    an[IllegalArgumentException] should be thrownBy {
      new OPSetTransformer[Text, Integral, MultiPickList, MultiPickList](
        transformer = new UnaryLambdaTransformer[Text, Integral](
          operationName = "unaryTextText",
          transformFn = (input: Text) => Integral(input.hashCode)
        ),
        operationName = "unaryTextTextWrap"
      )
    }

    an[IllegalArgumentException] should be thrownBy {
      new OPSetTransformer[Real, Text, MultiPickList, MultiPickList](
        transformer = new UnaryLambdaTransformer[Real, Text](
          operationName = "unaryRealText",
          transformFn = (input: Real) => FeatureTypeDefaults.Text
        ),
        operationName = "unaryRealTextWrap"
      )
    }
  }

  it should "fail to create collection transformers from a UnaryTransformer that already involves collections" in {
    an[IllegalArgumentException] should be thrownBy {
      new OPSetTransformer[MultiPickList, MultiPickList, MultiPickList, MultiPickList](
        transformer = new UnaryLambdaTransformer[MultiPickList, MultiPickList](
          operationName = "unaryMPMP",
          transformFn = (input: MultiPickList) => input
        ),
        operationName = "unaryMPMPWrap"
      )
    }

    an[IllegalArgumentException] should be thrownBy {
      new OPMapTransformer[RealMap, TextMap, RealMap, PhoneMap](
        transformer = new UnaryLambdaTransformer[RealMap, TextMap](
          operationName = "unaryRTRP",
          transformFn = (input: RealMap) => FeatureTypeDefaults.TextMap
        ),
        operationName = "unaryRTRPWrap"
      )
    }

    an[IllegalArgumentException] should be thrownBy {
      new OPListTransformer[TextList, DateTimeList, TextList, DateList](
        transformer = new UnaryLambdaTransformer[TextList, DateTimeList](
          operationName = "unaryRTRP",
          transformFn = (input: TextList) => FeatureTypeDefaults.DateTimeList
        ),
        operationName = "unaryRTRPWrap"
      )
    }
  }
}

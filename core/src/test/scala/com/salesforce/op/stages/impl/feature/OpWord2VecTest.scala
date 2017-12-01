/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class OpWord2VecTest extends FlatSpec with TestSparkContext {

  val data = Seq(
    "I I I like like Spark".split(" "),
    "Hi I heard about Spark".split(" "),
    "I wish Java could use case classes".split(" "),
    "Logistic regression models are neat".split(" ")
  ).map(_.toSeq.toTextList)

  lazy val (inputData, f1) = TestFeatureBuilder(Seq(data.head))
  lazy val (testData, _) = TestFeatureBuilder(data.tail)

  lazy val expected = data.tail.zip(Seq(
    Vectors.dense(-0.029884086549282075, -0.055613189935684204, 0.04186216294765473).toOPVector,
    Vectors.dense(-0.0026281912411962234, -0.016138136386871338, 0.010740748473576136).toOPVector,
    Vectors.dense(0.0, 0.0, 0.0).toOPVector
  )).toArray

  Spec[OpWord2VecTest] should "convert array of strings into a vector" in {
    val f1Vec = new OpWord2Vec().setInput(f1).setMinCount(0).setVectorSize(3).setSeed(1234567890L)
    val output = f1Vec.getOutput()
    val testTransformedData = f1Vec.fit(inputData).transform(testData)
    testTransformedData.orderBy(f1.name).collect(f1, output) shouldBe expected
  }

  it should "convert array of strings into a vector (shortcut version)" in {
    val output = f1.word2vec(minCount = 0, vectorSize = 3)
    val f1Vec = output.originStage.asInstanceOf[OpWord2Vec].setSeed(1234567890L)
    val testTransformedData = f1Vec.fit(inputData).transform(testData)
    testTransformedData.orderBy(f1.name).collect(f1, output) shouldBe expected
  }

}

/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.test.PassengerSparkFixtureTest
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class VectorsCombinerTest extends FlatSpec with PassengerSparkFixtureTest {

  val vectors = Seq(
    Vectors.sparse(4, Array(0, 3), Array(1.0, 1.0)),
    Vectors.dense(Array(2.0, 3.0, 4.0)),
    Vectors.sparse(4, Array(1), Array(777.0))
  )
  val expected = Vectors.sparse(11, Array(0, 3, 4, 5, 6, 8), Array(1.0, 1.0, 2.0, 3.0, 4.0, 777.0))

  Spec[VectorsCombiner] should "combine vectors correctly" in {
    val combined = VectorsCombiner.combine(vectors)
    assert(combined.compressed == combined, "combined is expected to be compressed")
    combined shouldBe expected
  }

}

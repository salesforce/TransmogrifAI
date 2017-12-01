/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.io.avro

import java.io.{File, FileWriter}

import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.io.avro.AvroInOut._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class AvroInOutTest extends FlatSpec with TestSparkContext {

  "AvroInOut" should "readPathSeq" in {
    // TODO
  }

  "AvroWriter" should "read" in {
    // TODO
  }

  it should "checkPathsExist" in {
    val f1 = new File("/tmp/avroinouttest")
    f1.delete()
    val w = new FileWriter(f1)
    w.write("just checking")
    w.close()
    val f2 = new File("/tmp/thisfilecannotexist")
    f2.delete()
    val f3 = new File("/tmp/this file cannot exist")
    f3.delete()
    assume(f1.exists && !f2.exists && !f3.exists)

    selectExistingPaths(s"$f1,$f2") shouldBe f1.toString

    intercept[IllegalArgumentException] { selectExistingPaths(f2.toString) }
    intercept[IllegalArgumentException] { selectExistingPaths(f3.toString) }

    selectExistingPaths(s"$f1,$f3") shouldBe f1.toString

    intercept[IllegalArgumentException] { selectExistingPaths("") }
  }

}
